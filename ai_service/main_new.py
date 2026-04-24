# ai_service/main_new.py
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pathlib import Path
import cv2
import numpy as np
import json
import time
import base64
from io import BytesIO
import shutil
import uuid
import redis
import logging
from typing import List, Optional, Dict, Tuple
from datetime import datetime

from detection_line import DetectionLine, parse_detection_line, default_vertical_line
from config import config
import os
import subprocess

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="交通流量分析系统 - 推理服务", version="3.0.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis连接
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Redis连接成功")
except Exception as e:
    logger.warning(f"Redis连接失败: {e}，将使用内存存储")
    redis_client = None

memory_storage = {}
active_tasks = {}  # 存储正在处理的任务

# 结果视频存储目录
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============ DeepSORT 跟踪器 ============

class SimpleDeepSORT:
    """简化版DeepSORT实现"""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1

    def update(self, detections: np.ndarray, frame: np.ndarray) -> List[Dict]:
        if len(detections) == 0:
            expired = []
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['time_since_update'] += 1
                if self.tracks[track_id]['time_since_update'] > self.max_age:
                    expired.append(track_id)

            for track_id in expired:
                del self.tracks[track_id]

            return self._get_confirmed_tracks()

        # 预测位置
        for track in self.tracks.values():
            if 'velocity' in track:
                track['bbox'][0] += track['velocity'][0]
                track['bbox'][1] += track['velocity'][1]
                track['bbox'][2] += track['velocity'][0]
                track['bbox'][3] += track['velocity'][1]

        # 计算IOU矩阵
        track_ids = list(self.tracks.keys())
        track_bboxes = np.array([self.tracks[tid]['bbox'] for tid in track_ids])
        iou_matrix = self._compute_iou(detections[:, :4], track_bboxes)

        # 贪婪匹配
        matched_det_indices = set()
        matched_track_ids = set()
        matches = []

        for i in range(len(detections)):
            for j in range(len(track_ids)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    matches.append((iou_matrix[i, j], i, track_ids[j]))

        matches.sort(reverse=True)

        for iou, det_idx, track_id in matches:
            if det_idx in matched_det_indices or track_id in matched_track_ids:
                continue

            old_bbox = self.tracks[track_id]['bbox']
            new_bbox = detections[det_idx, :4].tolist()
            velocity = [
                (new_bbox[0] + new_bbox[2] - old_bbox[0] - old_bbox[2]) / 2,
                (new_bbox[1] + new_bbox[3] - old_bbox[1] - old_bbox[3]) / 2
            ]

            self.tracks[track_id].update({
                'bbox': new_bbox,
                'score': float(detections[det_idx, 4]),
                'hits': self.tracks[track_id].get('hits', 0) + 1,
                'time_since_update': 0,
                'velocity': velocity
            })

            matched_det_indices.add(det_idx)
            matched_track_ids.add(track_id)

        # 新跟踪器
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                self.tracks[self.next_id] = {
                    'bbox': det[:4].tolist(),
                    'score': float(det[4]),
                    'hits': 1,
                    'time_since_update': 0,
                    'velocity': [0, 0]
                }
                self.next_id += 1

        # 未匹配的标记为丢失
        for track_id in track_ids:
            if track_id not in matched_track_ids:
                self.tracks[track_id]['time_since_update'] += 1
                if self.tracks[track_id]['time_since_update'] > self.max_age:
                    del self.tracks[track_id]

        return self._get_confirmed_tracks()

    def _compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        if len(boxes2) == 0:
            return np.zeros((len(boxes1), 0))

        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - inter

        return inter / (union + 1e-6)

    def _get_confirmed_tracks(self) -> List[Dict]:
        results = []
        for track_id, track in self.tracks.items():
            if track.get('hits', 0) >= self.min_hits:
                results.append({
                    'track_id': track_id,
                    'bbox': track['bbox'],
                    'score': track['score']
                })
        return results


# ============ 虚拟检测线和流量统计 ============

class TrafficCounter:
    def __init__(self, lines: List[DetectionLine], frame_width: int, frame_height: int):
        self.lines = lines
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.counts = {line.name: {'in': 0, 'out': 0} for line in lines}
        self.vehicle_history = {}
        self.violations = []
        self.stationary_frames = {}

    def update(self, tracks: List[Dict], frame_idx: int, timestamp: float):
        current_tracks = {}

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            current_tracks[track_id] = {'center': center, 'bbox': bbox}

            # 检查跨越检测线
            if track_id in self.vehicle_history:
                last_pos = self.vehicle_history[track_id]['center']

                for line in self.lines:
                    line_px = (
                        line.x1 * self.frame_width,
                        line.y1 * self.frame_height,
                        line.x2 * self.frame_width,
                        line.y2 * self.frame_height
                    )

                    if self._crosses_line(last_pos, center, line_px):
                        direction = self._determine_direction(last_pos, center, line)

                        if direction == 'in':
                            self.counts[line.name]['in'] += 1
                        else:
                            self.counts[line.name]['out'] += 1

                            # 逆行检测
                            if line.direction == 'vertical' and center[1] < last_pos[1]:
                                self.violations.append({
                                    'type': 'wrong_direction',
                                    'track_id': track_id,
                                    'timestamp': timestamp,
                                    'frame': frame_idx,
                                    'location': center
                                })

            # 违停检测
            if track_id in self.stationary_frames:
                info = self.stationary_frames[track_id]
                dx = center[0] - info['position'][0]
                dy = center[1] - info['position'][1]
                distance = np.sqrt(dx * dx + dy * dy)

                if distance < 5:
                    info['frames'] += 1
                    if info['frames'] > 90:
                        self.violations.append({
                            'type': 'illegal_parking',
                            'track_id': track_id,
                            'timestamp': timestamp,
                            'frame': frame_idx,
                            'location': center
                        })
                        info['frames'] = 0
                else:
                    info['position'] = center
                    info['frames'] = 0
            else:
                self.stationary_frames[track_id] = {'position': center, 'frames': 0}

        self.vehicle_history = current_tracks

        return {
            'counts': self.counts,
            'active_vehicles': len(current_tracks),
            'total_violations': len(self.violations)
        }

    def _crosses_line(self, p1, p2, line):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = (line[0], line[1]), (line[2], line[3])
        C, D = p1, p2
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def _determine_direction(self, p1, p2, line):
        if line.direction == 'vertical':
            return 'in' if p2[0] > p1[0] else 'out'
        else:
            return 'in' if p2[1] > p1[1] else 'out'


# ============ 可视化绘制工具 ============

class ResultVisualizer:
    """检测结果可视化"""

    # ✅ 关键修改：更新为 UA-DETRAC 4 类颜色
    COLORS = {
        'car': (0, 255, 0),  # 绿色
        'bus': (255, 165, 0),  # 橙色
        'van': (0, 0, 255),  # 红色
        'others': (128, 128, 128),  # 灰色
        'default': (255, 255, 0)  # 青色（备用）
    }

    @staticmethod
    def draw_detection(frame: np.ndarray, track: Dict, color: Tuple[int, int, int] = None):
        """绘制单个检测结果"""
        bbox = track['bbox']
        track_id = track.get('track_id', 0)
        score = track.get('score', 0)

        # ✅ 修改：如果有 class_name 使用对应颜色，否则用传入颜色或默认
        class_name = track.get('class_name', 'default')
        if color is None:
            color = ResultVisualizer.COLORS.get(class_name, ResultVisualizer.COLORS['default'])

        x1, y1, x2, y2 = map(int, bbox)

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制ID标签（包含类别名）
        label = f"{class_name} ID:{track_id} {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    @staticmethod
    def draw_detection_line(frame: np.ndarray, line: DetectionLine,
                            width: int, height: int, color: Tuple[int, int, int] = (0, 0, 255)):
        """绘制检测线"""
        x1 = int(line.x1 * width)
        y1 = int(line.y1 * height)
        x2 = int(line.x2 * width)
        y2 = int(line.y2 * height)

        cv2.line(frame, (x1, y1), (x2, y2), color, 3)

        # 绘制标签
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(frame, line.name, (mid_x + 10, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    @staticmethod
    def draw_stats(frame: np.ndarray, stats: Dict, counter: TrafficCounter,
                   frame_idx: int, fps: float):
        """绘制统计信息"""
        h, w = frame.shape[:2]

        # 创建半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)  # 加宽以显示4类统计
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 绘制文字
        y_offset = 40
        cv2.putText(frame, f"Frame: {frame_idx}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_offset += 30
        cv2.putText(frame, f"Active Vehicles: {stats['active_vehicles']}",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y_offset += 30
        cv2.putText(frame, f"Violations: {stats['total_violations']}",
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 绘制流量统计
        y_offset += 30
        for line_name, counts in stats['counts'].items():
            text = f"{line_name}: In={counts['in']} Out={counts['out']}"
            cv2.putText(frame, text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25

        return frame


# ============ 检测器 ============

class SimpleDetector:
    """简化版检测器（测试模式）"""

    def __init__(self):
        logger.info("初始化简化版检测器（测试模式）")
        self.frame_count = 0
        # ✅ 修改为4类
        self.class_names = ['car', 'bus', 'van', 'others']

    def predict(self, frame: np.ndarray):
        import random
        time.sleep(0.01)  # 模拟延迟

        h, w = frame.shape[:2]
        num_detections = random.randint(1, 5)

        detections = []
        for i in range(num_detections):
            x1 = random.randint(50, w - 200)
            y1 = random.randint(50, h - 200)
            w_box = random.randint(80, 150)
            h_box = random.randint(60, 120)

            detections.append({
                'class_name': random.choice(self.class_names),  # ✅ 从4类中随机
                'score': random.uniform(0.6, 0.95),
                'bbox': [float(x1), float(y1), float(x1 + w_box), float(y1 + h_box)]
            })

        return detections, 0.01


# ✅ 关键修改：初始化新的4类检测器
try:
    # 确保 test_inference_new.py 在 Python 路径中
    from test_inference_new import RTDETRPredictor

    detector = RTDETRPredictor(
        config.MODEL_PATH,
        use_onnx=True,
        conf_threshold=0.35,
        nms_threshold=0.5
    )
    logger.info("✅ RT-DETR UA-DETRAC 4类模型加载成功")
except Exception as e:
    logger.warning(f"⚠️ 无法加载RT-DETR模型: {e}，使用测试模式")
    detector = SimpleDetector()

tracker = SimpleDeepSORT(max_age=30, min_hits=3)
visualizer = ResultVisualizer()


# ============ 任务状态管理 ============

def update_task_status(task_id: str, status: str, progress: int, message: str,
                       result: Optional[Dict] = None, error: Optional[str] = None):
    data = {
        'status': status,
        'progress': progress,
        'message': message,
        'update_time': time.time()
    }
    if result:
        data['result'] = json.dumps(result)
    if error:
        data['error'] = error

    if redis_client:
        redis_client.hset(f"task:{task_id}", mapping=data)
        redis_client.expire(f"task:{task_id}", 86400)
    else:
        memory_storage[task_id] = data


def get_task_status(task_id: str) -> Optional[Dict]:
    if redis_client:
        data = redis_client.hgetall(f"task:{task_id}")
        return data if data else None
    else:
        return memory_storage.get(task_id)


# ============ 视频处理任务 ============

def process_video_task(task_id: str, video_path: Path, frame_skip: int = 3,
                       detection_lines: Optional[List[Dict]] = None):
    """处理视频并生成结果视频（支持4类检测）"""
    cap = None
    out = None
    is_cancelled = False
    temp_output = None

    try:
        logger.info(f"任务 {task_id}: 开始处理 {video_path}")
        update_task_status(task_id, "processing", 5, "初始化视频...")

        # 安全处理 detection_lines
        lines_data = []
        if detection_lines and isinstance(detection_lines, list):
            lines_data = detection_lines
        elif detection_lines and isinstance(detection_lines, str):
            try:
                parsed = json.loads(detection_lines)
                if isinstance(parsed, list):
                    lines_data = parsed
            except:
                logger.warning(f"解析 detection_lines 失败，使用默认值")

        if not lines_data:
            lines_data = [{
                'name': 'main_line',
                'x1': 0.5, 'y1': 0.0,
                'x2': 0.5, 'y2': 1.0,
                'direction': 'vertical'
            }]

        validated_lines = []
        for line in lines_data:
            pl = parse_detection_line(line)
            if pl is not None:
                validated_lines.append(pl)
            else:
                logger.warning(f"跳过无效的检测线配置: {line}")

        if not validated_lines:
            validated_lines = [default_vertical_line()]

        # 打开输入视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

        # 最终输出路径
        final_output = RESULTS_DIR / f"{task_id}_result.mp4"
        temp_output = RESULTS_DIR / f"{task_id}_temp.mp4"

        # 清理旧文件
        if final_output.exists():
            final_output.unlink()
        if temp_output.exists():
            temp_output.unlink()

        # 先用 OpenCV 创建视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps / frame_skip, (width, height))

        if not out.isOpened():
            raise ValueError("无法创建临时视频文件")

        lines = validated_lines
        counter = TrafficCounter(lines, width, height)

        frame_results = []
        frame_count = 0
        processed_count = 0
        total_detections = 0

        # ✅ 新增：4类统计
        class_stats = {'car': 0, 'bus': 0, 'van': 0, 'others': 0}

        update_task_status(task_id, "processing", 10, "开始分析视频...")

        while True:
            if task_id not in active_tasks:
                logger.info(f"任务 {task_id}: 任务已被取消，停止处理")
                is_cancelled = True
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            timestamp = frame_count / fps

            # 检测
            try:
                detections, infer_time = detector.predict(frame)
            except Exception as e:
                logger.error(f"检测失败 frame {frame_count}: {e}")
                detections = []
                infer_time = 0

            total_detections += len(detections)

            # ✅ 统计各类别
            for det in detections:
                cls_name = det.get('class_name', 'unknown')
                if cls_name in class_stats:
                    class_stats[cls_name] += 1

            # 跟踪（跟踪器只需要bbox和score，不管类别）
            detection_list = []
            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                detection_list.append([x1, y1, x2, y2, d['score']])

            if len(detection_list) > 0:
                np_detections = np.array(detection_list)
                tracks = tracker.update(np_detections, frame)

                # ✅ 将类别信息附加到tracks用于可视化
                for track in tracks:
                    # 找到匹配的检测框，获取类别
                    min_dist = float('inf')
                    matched_class = 'car'
                    for det in detections:
                        # 计算中心点距离
                        det_cx = (det['bbox'][0] + det['bbox'][2]) / 2
                        det_cy = (det['bbox'][1] + det['bbox'][3]) / 2
                        track_cx = (track['bbox'][0] + track['bbox'][2]) / 2
                        track_cy = (track['bbox'][1] + track['bbox'][3]) / 2
                        dist = np.sqrt((det_cx - track_cx) ** 2 + (det_cy - track_cy) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            matched_class = det['class_name']
                    track['class_name'] = matched_class
            else:
                tracks = []

            # 流量统计
            stats = counter.update(tracks, frame_count, timestamp)

            # 可视化
            vis_frame = frame.copy()
            for line in lines:
                visualizer.draw_detection_line(vis_frame, line, width, height)
            for track in tracks:
                visualizer.draw_detection(vis_frame, track)  # 现在会显示类别颜色
            visualizer.draw_stats(vis_frame, stats, counter, frame_count, fps)

            # 写入临时文件
            out.write(vis_frame)

            # 保存结果数据
            frame_data = {
                'frame': frame_count,
                'timestamp': round(timestamp, 2),
                'count': len(detections),
                'tracks': tracks,
                'detections': detections,
                'traffic_stats': stats,
                'infer_ms': round(infer_time * 1000, 1)
            }
            frame_results.append(frame_data)
            processed_count += 1

            # 更新进度
            if processed_count % 30 == 0:
                if task_id not in active_tasks:
                    is_cancelled = True
                    break

                progress = min(95, int(frame_count / total_frames * 100)) if total_frames > 0 else 0
                class_dist_str = ", ".join([f"{k}:{v}" for k, v in class_stats.items()])
                update_task_status(
                    task_id, "processing", progress,
                    f"处理中... {frame_count}/{total_frames}帧, 类别统计: {class_dist_str}"
                )

        # 释放资源
        if out:
            out.release()
            out = None
        if cap:
            cap.release()
            cap = None

        # 处理取消逻辑
        if is_cancelled:
            logger.info(f"任务 {task_id}: 正在清理取消的任务...")
            if temp_output.exists():
                temp_output.unlink(missing_ok=True)
            return

        # FFmpeg 转码（浏览器兼容）
        if temp_output.exists() and temp_output.stat().st_size > 0:
            update_task_status(task_id, "processing", 96, "正在转码视频为浏览器兼容格式...")

            ffmpeg_cmd = r"D:\ffmpeg\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
            if ffmpeg_cmd:
                try:
                    cmd = [
                        ffmpeg_cmd, '-i', str(temp_output),
                        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                        '-profile:v', 'baseline', '-level', '3.0',
                        '-preset', 'fast', '-crf', '23',
                        '-movflags', '+faststart', '-an', '-y',
                        str(final_output)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                    if result.returncode == 0 and final_output.exists():
                        logger.info(f"✅ FFmpeg 转码成功")
                        temp_output.unlink(missing_ok=True)
                    else:
                        logger.error(f"FFmpeg 失败: {result.stderr}")
                        temp_output.rename(final_output)
                except Exception as e:
                    logger.error(f"FFmpeg 异常: {e}")
                    temp_output.rename(final_output)
            else:
                logger.warning("未找到 FFmpeg，使用原始编码")
                temp_output.rename(final_output)
        else:
            raise ValueError("临时视频文件生成失败")

        # 验证最终输出
        if not final_output.exists() or final_output.stat().st_size == 0:
            raise ValueError("最终视频文件生成失败")

        # 汇总结果
        final_result = {
            'video_info': {
                'total_frames': frame_count,
                'fps': fps / frame_skip,
                'duration_sec': round(frame_count / fps, 1),
                'width': width,
                'height': height
            },
            'statistics': {
                'total_detections': total_detections,
                'avg_cars_per_frame': round(total_detections / processed_count, 1) if processed_count > 0 else 0,
                'processed_frames': processed_count,
                'unique_vehicles': len(counter.vehicle_history),
                'traffic_counts': counter.counts,
                'class_distribution': class_stats,  # ✅ 新增：4类分布统计
                'violations': {
                    'total': len(counter.violations),
                    'details': counter.violations[:50]
                }
            },
            'output_files': {
                'result_video': str(final_output),
                'result_video_url': f"/api/analyze/video/{task_id}"
            },
            'frame_results': frame_results[:100]
        }

        update_task_status(task_id, "completed", 100, "分析完成！", result=final_result)
        logger.info(f"任务 {task_id}: 完成！类别统计: {class_stats}")

    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {str(e)}")
        update_task_status(task_id, "failed", -1, f"失败: {str(e)}", error=str(e))
        is_cancelled = True

    finally:
        if out:
            out.release()
        if cap:
            cap.release()
        if temp_output and temp_output.exists():
            try:
                temp_output.unlink(missing_ok=True)
            except:
                pass
        if task_id in active_tasks:
            del active_tasks[task_id]
        try:
            if video_path.exists():
                video_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"删除输入文件失败: {e}")


# ============ API端点 ============

@app.post("/api/analyze/upload")
async def upload_video(
        background_tasks: BackgroundTasks,
        task_id: str = Form(...),
        file: UploadFile = File(...),
        frame_skip: int = Form(3),
        detection_lines: Optional[str] = Form(None)
):
    """上传视频并启动分析"""
    try:
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        original_name = file.filename or "unknown.mp4"
        file_ext = Path(original_name).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(400, f"不支持的文件格式: {file_ext}")

        if frame_skip < 1 or frame_skip > 10:
            raise HTTPException(400, "frame_skip必须在1-10之间")

        # 保存文件
        temp_dir = Path("temp/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{task_id}{file_ext}"

        logger.info(f"保存上传文件: {video_path}")

        active_tasks[task_id] = {
            "status": "processing",
            "start_time": time.time(),
            "video_path": str(video_path)
        }

        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 解析检测线
        lines = None
        if detection_lines:
            try:
                lines = json.loads(detection_lines)
            except json.JSONDecodeError:
                raise HTTPException(400, "detection_lines格式错误")

        update_task_status(task_id, "pending", 0, "视频上传成功，等待处理")

        background_tasks.add_task(process_video_task, task_id, video_path, frame_skip, lines)

        return JSONResponse({
            "code": 200,
            "message": "视频上传成功，开始分析",
            "data": {
                "task_id": task_id,
                "status": "processing",
                "check_url": f"/api/analyze/result/{task_id}",
                "video_url": f"/api/analyze/video/{task_id}"
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传失败: {str(e)}")
        raise HTTPException(500, str(e))


@app.get("/api/analyze/result/{task_id}")
async def get_result(task_id: str):
    """获取分析结果"""
    try:
        task_data = get_task_status(task_id)

        if not task_data:
            return JSONResponse({
                "code": 404,
                "message": "任务不存在",
                "data": None
            })

        result = None
        if task_data.get('result'):
            if isinstance(task_data['result'], str):
                try:
                    result = json.loads(task_data['result'])
                except:
                    result = task_data['result']
            else:
                result = task_data['result']

        return JSONResponse({
            "code": 200,
            "data": {
                "task_id": task_id,
                "status": task_data.get('status'),
                "progress": int(task_data.get('progress', 0)),
                "message": task_data.get('message'),
                "result": result,
                "error": task_data.get('error')
            }
        })

    except Exception as e:
        raise HTTPException(500, str(e))


@app.api_route("/api/analyze/video/{task_id}", methods=["GET", "HEAD"])
async def get_result_video(task_id: str, request: Request, download: bool = False):
    """获取结果视频"""
    try:
        video_path = RESULTS_DIR / f"{task_id}_result.mp4"

        if not video_path.exists():
            task_data = get_task_status(task_id)
            if not task_data:
                raise HTTPException(404, "任务不存在")

            status = task_data.get('status')
            if status == 'processing':
                raise HTTPException(202, "视频正在处理中，请稍后")
            elif status == 'failed':
                raise HTTPException(500, "视频处理失败")
            else:
                raise HTTPException(404, "结果视频不存在")

        if request.method == "HEAD":
            file_size = video_path.stat().st_size
            headers = {
                "Content-Type": "video/mp4",
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Last-Modified": datetime.fromtimestamp(video_path.stat().st_mtime).strftime(
                    "%a, %d %b %Y %H:%M:%S GMT")
            }
            return Response(headers=headers)

        if download:
            return FileResponse(
                path=video_path,
                media_type="video/mp4",
                filename=f"{task_id}_result.mp4"
            )
        else:
            return FileResponse(
                path=video_path,
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f"inline; filename={task_id}_result.mp4",
                    "Accept-Ranges": "bytes"
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/analyze/videos")
async def list_result_videos():
    """列出所有结果视频"""
    try:
        videos = []
        for video_file in RESULTS_DIR.glob("*_result.mp4"):
            task_id = video_file.stem.replace("_result", "")
            stat = video_file.stat()
            videos.append({
                "task_id": task_id,
                "filename": video_file.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "url": f"/api/analyze/video/{task_id}"
            })

        return JSONResponse({
            "code": 200,
            "data": {
                "total": len(videos),
                "videos": sorted(videos, key=lambda x: x['created'], reverse=True)
            }
        })

    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/analyze/video/{task_id}")
async def delete_result_video(task_id: str):
    """删除结果视频"""
    try:
        video_path = RESULTS_DIR / f"{task_id}_result.mp4"

        if video_path.exists():
            video_path.unlink()
            return JSONResponse({
                "code": 200,
                "message": "视频已删除"
            })
        else:
            raise HTTPException(404, "视频不存在")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/analyze/frame")
async def analyze_single_frame(file: UploadFile = File(...)):
    """单帧图片检测，返回带标记框的图片（支持4类）"""
    try:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        file_ext = Path(file.filename or "").suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(400, f"不支持的图片格式: {file_ext}")

        # 读取图片数据
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(400, "无法读取图片")

        h, w = frame.shape[:2]

        # 检测
        start_time = time.time()
        detections, _ = detector.predict(frame)
        infer_time = time.time() - start_time

        # 绘制结果
        vis_frame = frame.copy()

        # ✅ 修改为4类颜色（与ResultVisualizer一致）
        colors = {
            'car': (0, 255, 0),  # 绿色
            'bus': (255, 165, 0),  # 橙色
            'van': (0, 0, 255),  # 红色
            'others': (128, 128, 128)  # 灰色
        }

        detection_results = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            score = det['score']
            color = colors.get(class_name, (128, 128, 128))

            # 绘制边界框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景
            label = f"{class_name} {score:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            label_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10

            cv2.rectangle(vis_frame,
                          (x1, label_y - text_h - 5),
                          (x1 + text_w, label_y + 5),
                          color, -1)

            # 绘制文字
            cv2.putText(vis_frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 保存检测结果
            detection_results.append({
                'id': i + 1,
                'class': class_name,
                'score': round(score, 3),
                'bbox': [x1, y1, x2, y2],
                'area': (x2 - x1) * (y2 - y1),
                'center': [(x1 + x2) // 2, (y1 + y2) // 2]
            })

        # 添加整体信息（显示4类统计）
        class_counts = {}
        for det in detection_results:
            cls = det['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        count_str = " ".join([f"{k}:{v}" for k, v in class_counts.items()])
        info_text = f"Total: {len(detections)} | {count_str} | {infer_time * 1000:.1f}ms"
        cv2.putText(vis_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 编码为base64
        _, buffer = cv2.imencode('.jpg', vis_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            "code": 200,
            "data": {
                "count": len(detections),
                "detections": detection_results,
                "infer_time_ms": round(infer_time * 1000, 1),
                "image_width": w,
                "image_height": h,
                "marked_image": f"data:image/jpeg;base64,{img_base64}",
                "statistics": {
                    "total": len(detections),
                    "by_class": class_counts
                }
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"单帧检测失败: {e}")
        raise HTTPException(500, str(e))


@app.post("/api/analyze/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消正在处理的任务"""
    if task_id in active_tasks:
        del active_tasks[task_id]
        update_task_status(task_id, "cancelled", -1, "用户已取消")
        return {"success": True}
    return {"success": False, "message": "任务不存在"}


@app.get("/health")
async def health_check():
    model_type = "rtdetr_detrac_4class" if not isinstance(detector, SimpleDetector) else "test_mode"
    return {
        "status": "ok",
        "model": model_type,
        "version": "3.0.0",
        "classes": ["car", "bus", "van", "others"],
        "redis": "connected" if redis_client else "disabled",
        "results_dir": str(RESULTS_DIR.absolute())
    }


@app.get("/api/analyze/video/{task_id}/check")
async def check_video(task_id: str):
    """检查视频文件状态"""
    video_path = RESULTS_DIR / f"{task_id}_result.mp4"

    return {
        "task_id": task_id,
        "file_exists": video_path.exists(),
        "file_path": str(video_path.absolute()),
        "file_size": video_path.stat().st_size if video_path.exists() else 0,
        "results_dir": str(RESULTS_DIR.absolute()),
        "files_in_results": [f.name for f in RESULTS_DIR.glob("*.mp4")]
    }


if __name__ == "__main__":
    import uvicorn

    Path("temp/uploads").mkdir(parents=True, exist_ok=True)
    Path("temp/results").mkdir(parents=True, exist_ok=True)

    logger.info(f"结果视频保存目录: {RESULTS_DIR.absolute()}")
    logger.info(f"当前使用模型: {'UA-DETRAC 4类' if not isinstance(detector, SimpleDetector) else '测试模式'}")

    uvicorn.run(app, host="0.0.0.0", port=8000)