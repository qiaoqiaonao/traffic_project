"""
交通流量分析系统 - AI服务（紧急修复版）
重点修复：状态保存可靠性，确保前端能拿到结果
"""
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
import sys
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import subprocess
from collections import deque, defaultdict

from detection_line import DetectionLine, parse_detection_line, default_vertical_line
from config import config

# ============ 日志配置 ============
Path("logs").mkdir(exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.handlers = []
logger.propagate = False

file_handler = logging.FileHandler("logs/ai_service.log", encoding='utf-8', mode='a')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
# ================================================================

app = FastAPI(title="交通流量分析系统 - 稳定版", version="3.2.1-fix")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============ 存储配置（双重保险） ============
# 1. 尝试Redis
redis_client = None
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True, socket_connect_timeout=2)
    redis_client.ping()
    logger.info("✅ Redis连接成功")
except Exception as e:
    logger.warning(f"⚠️ Redis连接失败: {e}，将使用内存存储")

# 2. 内存存储（备用）
memory_storage = {}

# 3. 持久化存储（终极保险，防止内存丢失）
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
status_backup_dir = Path("status_backup")
status_backup_dir.mkdir(exist_ok=True)

active_tasks = {}

# ============ 优化开关（当前只开图像增强） ============
ENABLE_IMAGE_ENHANCE = True
logger.info(f"配置: 图像增强={ENABLE_IMAGE_ENHANCE}")

REDIS_ENABLED = True

# ============ 可靠的存储函数（双重保险） ============
def save_task_status(task_id: str, status: str, progress: int, message: str,
                     result: Optional[Dict] = None, error: Optional[str] = None) -> bool:
    """
    三重保险保存：Redis + 内存 + 文件
    关键修复：Redis 写入失败时，清理旧 Key 防止后续脏读
    """
    data = {
        'status': status,
        'progress': progress,
        'message': message,
        'update_time': time.time()
    }
    if result is not None:
        try:
            data['result'] = json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"序列化 result 失败 [{task_id}]: {e}")
            data['result_error'] = f"序列化失败: {e}"
    if error:
        data['error'] = error

    success_flags = []

    # 1. Redis（带清理逻辑）
    if REDIS_ENABLED and redis_client:
        try:
            # 关键修复：先删除旧 hash，再重新写入，避免残留字段
            redis_client.delete(f"task:{task_id}")
            redis_client.hset(f"task:{task_id}", mapping=data)
            redis_client.expire(f"task:{task_id}", 86400)
            logger.info(f"💾 Redis保存成功 [{task_id}]: {status}")
            success_flags.append("redis")
        except Exception as e:
            logger.error(f"❌ Redis保存失败 [{task_id}]: {e}")

    # 2. 内存（无论 Redis 是否成功，都写入，确保本进程内一定可见）
    try:
        memory_storage[task_id] = data.copy()
        logger.info(f"💾 内存保存成功 [{task_id}]: {status}")
        success_flags.append("memory")
    except Exception as e:
        logger.error(f"❌ 内存保存失败 [{task_id}]: {e}")

    # 3. 文件（终极保险）
    try:
        backup_file = status_backup_dir / f"{task_id}.json"
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 文件备份成功 [{task_id}]")
        success_flags.append("file")
    except Exception as e:
        logger.error(f"❌ 文件备份失败 [{task_id}]: {e}")

    if not success_flags:
        logger.critical(f"🆘 所有存储方式失败 [{task_id}]！")
        return False

    logger.info(f"✅ 状态已持久化 [{task_id}] → {success_flags}")
    return True


def load_task_status(task_id: str) -> Optional[Dict]:
    """
    关键修复：读取所有可用源，按 update_time 取最新，彻底避免 Redis 脏读
    """
    candidates = []

    # 1. 读取 Redis
    if REDIS_ENABLED and redis_client:
        try:
            redis_data = redis_client.hgetall(f"task:{task_id}")
            if redis_data and redis_data.get('status'):
                redis_data['_src'] = 'redis'
                candidates.append(redis_data)
                logger.debug(f"📂 Redis读取 [{task_id}]: {redis_data.get('status')}")
        except Exception as e:
            logger.warning(f"⚠️ Redis读取失败 [{task_id}]: {e}")

    # 2. 读取内存
    if task_id in memory_storage:
        mem_data = memory_storage[task_id].copy()
        mem_data['_src'] = 'memory'
        candidates.append(mem_data)
        logger.debug(f"📂 内存读取 [{task_id}]: {mem_data.get('status')}")

    # 3. 读取文件
    backup_file = status_backup_dir / f"{task_id}.json"
    if backup_file.exists():
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            file_data['_src'] = 'file'
            candidates.append(file_data)
            logger.debug(f"📂 文件读取 [{task_id}]: {file_data.get('status')}")
        except Exception as e:
            logger.error(f"❌ 文件读取失败 [{task_id}]: {e}")

    if not candidates:
        logger.warning(f"❌ 状态获取失败 [{task_id}]: 所有来源都找不到")
        return None

    # 关键修复：按 update_time 选最新的（处理字符串/浮点两种类型）
    def _get_time(item):
        t = item.get('update_time', 0)
        if isinstance(t, str):
            try:
                return float(t)
            except:
                return 0.0
        return float(t) if t else 0.0

    latest = max(candidates, key=_get_time)

    # 清理内部标记字段，避免污染返回
    latest.pop('_src', None)

    logger.info(
        f"✅ 状态获取成功 [{task_id}] 来源={latest.pop('_src', 'unknown')}: "
        f"status={latest.get('status')}, progress={latest.get('progress')}, "
        f"has_result={'result' in latest}"
    )
    return latest


# ============ 图像增强模块 ============
class ImageEnhancer:
    def __init__(self, enable=True):
        self.enable = enable
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance(self, frame):
        if not self.enable:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.GaussianBlur(enhanced, (3, 3), 0.5)

    def get_adaptive_conf(self, frame, base_conf=0.25):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 50: return base_conf * 0.8
        elif brightness > 200: return base_conf * 0.9
        return base_conf

    def get_lighting(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b = np.mean(gray)
        if b < 50: return "night"
        elif b > 200: return "strong_light"
        return "normal"

enhancer = ImageEnhancer(enable=ENABLE_IMAGE_ENHANCE)

# ============ 标准版DeepSORT（最稳定） ============
class SimpleDeepSORT:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections: np.ndarray, frame: np.ndarray) -> List[Dict]:
        if len(detections) == 0:
            expired = [tid for tid in list(self.tracks.keys())
                      if self.tracks[tid]['time_since_update'] > self.max_age]
            for tid in expired:
                del self.tracks[tid]
            return [t for t in self.tracks.values() if t.get('hits',0) >= self.min_hits]

        track_ids = list(self.tracks.keys())
        if len(track_ids) == 0:
            for det in detections:
                self.tracks[self.next_id] = {
                    'track_id': self.next_id, 'bbox': det[:4].tolist(),
                    'score': float(det[4]), 'hits': 1, 'time_since_update': 0,
                    'class_name': 'car'
                }
                self.next_id += 1
        else:
            track_bboxes = np.array([self.tracks[tid]['bbox'] for tid in track_ids])
            iou_matrix = self._compute_iou(detections[:, :4], track_bboxes)

            matched_det = set()
            matched_track = set()

            for i in range(len(detections)):
                best_iou, best_j = 0, -1
                for j in range(len(track_ids)):
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_j = j

                if best_iou >= self.iou_threshold and best_j not in matched_track:
                    tid = track_ids[best_j]
                    self.tracks[tid].update({
                        'bbox': detections[i, :4].tolist(),
                        'score': float(detections[i, 4]),
                        'hits': self.tracks[tid].get('hits', 0) + 1,
                        'time_since_update': 0
                    })
                    matched_det.add(i)
                    matched_track.add(best_j)

            for i in range(len(detections)):
                if i not in matched_det:
                    self.tracks[self.next_id] = {
                        'track_id': self.next_id, 'bbox': detections[i, :4].tolist(),
                        'score': float(detections[i, 4]), 'hits': 1, 'time_since_update': 0,
                        'class_name': 'car'
                    }
                    self.next_id += 1

            for j, tid in enumerate(track_ids):
                if j not in matched_track:
                    self.tracks[tid]['time_since_update'] += 1
                    if self.tracks[tid]['time_since_update'] > self.max_age:
                        del self.tracks[tid]

        return [t for t in self.tracks.values() if t.get('hits', 0) >= self.min_hits]

    def _compute_iou(self, boxes1, boxes2):
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

# ============ 其他类 ============
class TrafficCounter:
    def __init__(self, lines, frame_width, frame_height):
        self.lines = lines; self.frame_width = frame_width; self.frame_height = frame_height
        self.counts = {l.name: {'in':0, 'out':0} for l in lines}
        self.vehicle_history = {}; self.violations = []; self.stationary_frames = {}
        self.track_classes = {}

    def update(self, tracks, frame_idx, timestamp):
        current_tracks = {}
        for track in tracks:
            tid = track['track_id']
            bbox = track['bbox']
            center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
            cls = track.get('class_name', 'car')
            if tid not in self.track_classes:
                self.track_classes[tid] = cls
            current_tracks[tid] = {'center': center, 'bbox': bbox, 'class': self.track_classes[tid]}

            if tid in self.vehicle_history:
                last_pos = self.vehicle_history[tid]['center']
                for line in self.lines:
                    line_px = (line.x1*self.frame_width, line.y1*self.frame_height,
                              line.x2*self.frame_width, line.y2*self.frame_height)
                    if self._crosses_line(last_pos, center, line_px):
                        direction = 'in' if (line.direction=='vertical' and center[0]>last_pos[0]) else 'out'
                        self.counts[line.name][direction] += 1
        self.vehicle_history = current_tracks
        return {'counts': self.counts, 'active_vehicles': len(current_tracks), 'total_violations': len(self.violations)}

    def _crosses_line(self, p1, p2, line):
        def ccw(A,B,C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
        A,B = (line[0],line[1]),(line[2],line[3])
        return ccw(A,p1,p2) != ccw(B,p1,p2) and ccw(A,B,p1) != ccw(A,B,p2)

class ResultVisualizer:
    COLORS = {'car': (0,255,0), 'bus': (255,165,0), 'van': (0,0,255), 'others': (128,128,128)}

    def draw_detection(self, frame, track):
        bbox = track['bbox']
        tid = track.get('track_id', 0)
        score = track.get('score', 0)
        cls = track.get('class_name', 'car')
        color = self.COLORS.get(cls, (128,128,128))
        x1,y1,x2,y2 = map(int, bbox)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        label = f"{cls} ID:{tid} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        return frame

    def draw_detection_line(self, frame, line, w, h):
        x1,y1,x2,y2 = int(line.x1*w), int(line.y1*h), int(line.x2*w), int(line.y2*h)
        cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 3)
        cv2.putText(frame, line.name, ((x1+x2)//2+10, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return frame

    def draw_stats(self, frame, stats, counter, frame_idx, fps, lighting="normal"):
        h,w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (400,200), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        y = 40
        for text in [f"Frame: {frame_idx}", f"Light: {lighting}", f"Active: {stats['active_vehicles']}", f"Vio: {stats['total_violations']}"]:
            cv2.putText(frame, text, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y += 25
        return frame

class SimpleDetector:
    def __init__(self):
        import random
        self.r = random
        self.class_names = ['car', 'bus', 'van', 'others']
    def predict(self, frame):
        time.sleep(0.05)
        h, w = frame.shape[:2]
        dets = []
        for _ in range(self.r.randint(1, 5)):
            x1 = self.r.randint(50, w-200)
            y1 = self.r.randint(50, h-200)
            dets.append({
                'class_name': self.r.choice(self.class_names),
                'score': self.r.uniform(0.6, 0.95),
                'bbox': [float(x1), float(y1), float(x1+100), float(y1+80)]
            })
        return dets, 0.05

# 初始化组件
tracker = SimpleDeepSORT(max_age=30, min_hits=3, iou_threshold=0.3)
visualizer = ResultVisualizer()

try:
    from test_inference_new import RTDETRPredictor
    detector = RTDETRPredictor(config.MODEL_PATH, use_onnx=True, conf_threshold=0.35, nms_threshold=0.5)
    logger.info("✅ RT-DETR模型加载成功")
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    detector = SimpleDetector()
    logger.warning("使用模拟检测器")

# ============ 核心处理流程（重点修复状态保存） ============
def process_video_task(task_id, video_path, frame_skip=3, detection_lines=None):
    cap = None; out = None; is_cancelled = False; temp_output = None

    # 重置跟踪器
    tracker.reset()

    try:
        logger.info(f"{'='*50}")
        logger.info(f"任务 {task_id}: 开始处理")
        logger.info(f"视频路径: {video_path}")

        # 初始状态保存（立即保存，确保前端能查到）
        save_task_status(task_id, "processing", 5, "初始化视频...")

        # 解析检测线
        lines_data = detection_lines if isinstance(detection_lines, list) else \
                     json.loads(detection_lines) if isinstance(detection_lines, str) else \
                     [{'name':'main_line', 'x1':0.5, 'y1':0, 'x2':0.5, 'y2':1, 'direction':'vertical'}]
        lines = [x for x in (parse_detection_line(l) for l in lines_data) if x is not None]
        if not lines:
            lines = [default_vertical_line()]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

        final_output = RESULTS_DIR / f"{task_id}_result.mp4"
        temp_output = RESULTS_DIR / f"{task_id}_temp.mp4"

        # 清理旧文件
        if final_output.exists(): final_output.unlink()
        if temp_output.exists(): temp_output.unlink()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps/frame_skip, (width, height))

        if not out.isOpened():
            raise ValueError("无法创建视频写入器")

        counter = TrafficCounter(lines, width, height)
        frame_results = []
        frame_count = 0; processed_count = 0; total_detections = 0
        class_stats = defaultdict(int)

        save_task_status(task_id, "processing", 10, "开始分析...")

        while True:
            if task_id not in active_tasks:
                logger.info(f"任务 {task_id}: 用户取消")
                is_cancelled = True
                break

            ret, frame = cap.read()
            if not ret:
                logger.info(f"任务 {task_id}: 视频读取完毕")
                break

            frame_count += 1
            if frame_count % frame_skip != 0: continue

            timestamp = frame_count / fps

            # 图像增强
            if ENABLE_IMAGE_ENHANCE:
                process_frame = enhancer.enhance(frame)
                if hasattr(detector, 'conf_threshold'):
                    original_conf = detector.conf_threshold
                    detector.conf_threshold = enhancer.get_adaptive_conf(frame, 0.25)
                lighting = enhancer.get_lighting(frame)
            else:
                process_frame = frame
                lighting = "normal"

            # 检测
            try:
                detections, infer_time = detector.predict(process_frame)
                if ENABLE_IMAGE_ENHANCE and hasattr(detector, 'conf_threshold'):
                    detector.conf_threshold = original_conf
            except Exception as e:
                logger.error(f"检测失败帧 {frame_count}: {e}")
                detections, infer_time = [], 0

            total_detections += len(detections)
            for det in detections:
                class_stats[det.get('class_name', 'unknown')] += 1

            # 跟踪
            det_array = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['score']] for d in detections])
            tracks = tracker.update(det_array, frame)

            # 补充类别信息
            for track in tracks:
                min_dist = float('inf'); matched_cls = 'car'
                t_cx = (track['bbox'][0] + track['bbox'][2])/2
                t_cy = (track['bbox'][1] + track['bbox'][3])/2
                for det in detections:
                    d_cx = (det['bbox'][0] + det['bbox'][2])/2
                    d_cy = (det['bbox'][1] + det['bbox'][3])/2
                    dist = (t_cx-d_cx)**2 + (t_cy-d_cy)**2
                    if dist < min_dist:
                        min_dist = dist; matched_cls = det.get('class_name', 'car')
                track['class_name'] = matched_cls

            # 统计
            stats = counter.update(tracks, frame_count, timestamp)

            # 可视化
            vis_frame = frame.copy()
            for line in lines: visualizer.draw_detection_line(vis_frame, line, width, height)
            for track in tracks: visualizer.draw_detection(vis_frame, track)
            visualizer.draw_stats(vis_frame, stats, counter, frame_count, fps/frame_skip, lighting)

            out.write(vis_frame)

            frame_results.append({
                'frame': frame_count, 'timestamp': round(timestamp,2), 'count': len(detections),
                'tracks': tracks, 'traffic_stats': stats, 'infer_ms': round(infer_time*1000, 1)
            })
            processed_count += 1

            # 进度更新
            if processed_count % 30 == 0:
                progress = min(95, int(frame_count/total_frames*100)) if total_frames > 0 else 0
                save_task_status(task_id, "processing", progress,
                               f"处理中... {frame_count}/{total_frames}帧 | 光照:{lighting}")

        # 释放资源
        if out: out.release(); out = None
        if cap: cap.release(); cap = None

        if is_cancelled:
            logger.info(f"任务 {task_id}: 清理取消的任务")
            if temp_output.exists(): temp_output.unlink()
            save_task_status(task_id, "cancelled", -1, "用户已取消")
            return

        # FFmpeg转码
        if temp_output.exists() and temp_output.stat().st_size > 0:
            save_task_status(task_id, "processing", 96, "转码视频...")
            ffmpeg = r"D:\ffmpeg\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

            if Path(ffmpeg).exists():
                try:
                    cmd = [ffmpeg, '-i', str(temp_output), '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                           '-preset', 'fast', '-crf', '23', '-movflags', '+faststart', '-an', '-y', str(final_output)]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode == 0 and final_output.exists():
                        logger.info(f"FFmpeg转码成功")
                        temp_output.unlink()
                    else:
                        logger.error(f"FFmpeg失败: {result.stderr[:200]}")
                        temp_output.rename(final_output)
                except Exception as e:
                    logger.error(f"FFmpeg异常: {e}")
                    temp_output.rename(final_output)
            else:
                logger.warning("未找到FFmpeg，使用原始编码")
                temp_output.rename(final_output)

        # 验证结果
        if not final_output.exists() or final_output.stat().st_size == 0:
            raise ValueError("最终视频文件生成失败")

        # 构建结果（简化结构，确保JSON可序列化）
        final_result = {
            'video_info': {
                'total_frames': int(frame_count),
                'fps': float(fps/frame_skip),
                'duration_sec': round(float(frame_count/fps), 1),
                'width': int(width),
                'height': int(height)
            },
            'statistics': {
                'total_detections': int(total_detections),
                'processed_frames': int(processed_count),
                'unique_vehicles': len(counter.vehicle_history),
                'traffic_counts': dict(counter.counts),
                'class_distribution': dict(class_stats),
                'violations': {'total': len(counter.violations)}
            },
            'output_files': {
                'result_video': str(final_output.absolute()),
                'result_video_url': f"/api/analyze/video/{task_id}"
            },
            'optimization_enabled': {
                'image_enhancement': ENABLE_IMAGE_ENHANCE
            }
        }

        # 关键：任务完成状态保存（三重保险）
        logger.info(f"任务 {task_id}: 准备保存完成状态")
        save_success = save_task_status(task_id, "completed", 100, "分析完成！", result=final_result)

        if save_success:
            logger.info(f"✅ 任务 {task_id} 完成并保存成功！")
        else:
            logger.critical(f"🆘 任务 {task_id} 完成但保存失败！")

        # 延迟清理active_tasks，确保状态保存完成
        if task_id in active_tasks:
            del active_tasks[task_id]
            logger.info(f"任务 {task_id}: 已从active_tasks移除")

        logger.info(f"{'='*50}")


    except Exception as e:

        logger.error(f"任务 {task_id} 失败: {str(e)}", exc_info=True)

        save_task_status(task_id, "failed", -1, f"失败: {str(e)}", error=str(e))


    finally:

        # 关键：确保资源一定释放，active_tasks 一定清理

        if out:
            out.release()

        if cap:
            cap.release()

        if temp_output and temp_output.exists():

            try:

                temp_output.unlink(missing_ok=True)

            except Exception as e:

                logger.warning(f"清理临时文件失败: {e}")

        # 无论成功失败，一定要从 active_tasks 移除，否则任务一直占着内存

        if task_id in active_tasks:
            del active_tasks[task_id]

            logger.info(f"任务 {task_id}: 已从active_tasks移除")

        logger.info(f"{'=' * 50}")

# ============ API端点（修复查询逻辑） ============
# 调试接口
@app.get("/api/analyze/debug/{task_id}")
async def debug_task_storage(task_id: str):
    """诊断接口：查看 Redis/内存/文件 各自存储了什么"""
    debug_info = {"task_id": task_id, "sources": {}}

    # Redis
    if REDIS_ENABLED and redis_client:
        try:
            r = redis_client.hgetall(f"task:{task_id}")
            debug_info["sources"]["redis"] = {
                "exists": bool(r),
                "status": r.get("status") if r else None,
                "has_result": "result" in r if r else False,
                "update_time": r.get("update_time") if r else None
            }
        except Exception as e:
            debug_info["sources"]["redis"] = {"error": str(e)}

    # Memory
    if task_id in memory_storage:
        m = memory_storage[task_id]
        debug_info["sources"]["memory"] = {
            "exists": True,
            "status": m.get("status"),
            "has_result": "result" in m,
            "update_time": m.get("update_time")
        }
    else:
        debug_info["sources"]["memory"] = {"exists": False}

    # File
    backup_file = status_backup_dir / f"{task_id}.json"
    if backup_file.exists():
        try:
            with open(backup_file, 'r', encoding='utf-8') as f:
                d = json.load(f)
            debug_info["sources"]["file"] = {
                "exists": True,
                "status": d.get("status"),
                "has_result": "result" in d,
                "update_time": d.get("update_time")
            }
        except Exception as e:
            debug_info["sources"]["file"] = {"exists": True, "error": str(e)}
    else:
        debug_info["sources"]["file"] = {"exists": False}

    return JSONResponse({"code": 200, "data": debug_info})




@app.post("/api/analyze/upload")
async def upload_video(background_tasks: BackgroundTasks, task_id: str = Form(...),
                      file: UploadFile = File(...), frame_skip: int = Form(3),
                      detection_lines: Optional[str] = Form(None)):
    try:
        ext = Path(file.filename or "unknown.mp4").suffix.lower()
        if ext not in {'.mp4', '.avi', '.mov', '.mkv'}:
            raise HTTPException(400, f"不支持的格式: {ext}")

        if frame_skip < 1 or frame_skip > 10:
            raise HTTPException(400, "frame_skip必须在1-10之间")

        video_path = Path("temp/uploads") / f"{task_id}{ext}"
        video_path.parent.mkdir(parents=True, exist_ok=True)

        # 立即注册到active_tasks
        active_tasks[task_id] = {
            "status": "processing",
            "start_time": time.time(),
            "video_path": str(video_path)
        }

        logger.info(f"接收上传 [{task_id}]: {file.filename}")

        with open(video_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 立即保存初始状态（确保前端能立即查询到）
        save_task_status(task_id, "pending", 0, "视频上传成功，等待处理")

        # 解析检测线
        lines = None
        if detection_lines:
            try:
                lines = json.loads(detection_lines)
                logger.info(f"检测线配置: {lines}")
            except:
                raise HTTPException(400, "detection_lines格式错误")

        # 启动后台任务
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
    except HTTPException: raise
    except Exception as e:
        logger.error(f"上传失败: {e}")
        raise HTTPException(500, str(e))

@app.get("/api/analyze/result/{task_id}")
async def get_result(task_id: str):
    """
    查询任务结果 - 使用双重保险加载
    """
    try:
        logger.info(f"查询请求 [{task_id}]")

        # 使用可靠的加载函数
        data = load_task_status(task_id)

        if not data:
            logger.warning(f"查询 [{task_id}]: 任务不存在")
            return JSONResponse({
                "code": 404,
                "message": "任务不存在",
                "data": None
            })

        # 解析result字段
        result = None
        if data.get('result'):
            result_str = data['result']
            if isinstance(result_str, str):
                try:
                    result = json.loads(result_str)
                except json.JSONDecodeError as e:
                    logger.error(f"解析result失败 [{task_id}]: {e}")
                    result = result_str  # 保留原始字符串
            else:
                result = result_str

        response_data = {
            "task_id": task_id,
            "status": data.get('status'),
            "progress": int(data.get('progress', 0)),
            "message": data.get('message'),
            "result": result,
            "error": data.get('error')
        }

        logger.info(f"查询 [{task_id}]: 返回 status={data.get('status')}, progress={data.get('progress')}")

        return JSONResponse({
            "code": 200,
            "data": response_data
        })

    except Exception as e:
        logger.error(f"查询异常 [{task_id}]: {e}", exc_info=True)
        raise HTTPException(500, f"查询失败: {str(e)}")

@app.api_route("/api/analyze/video/{task_id}", methods=["GET", "HEAD"])
async def get_video(task_id: str, request: Request, download: bool = False):
    try:
        video_path = RESULTS_DIR / f"{task_id}_result.mp4"

        if not video_path.exists():
            # 检查状态
            data = load_task_status(task_id)
            if not data:
                raise HTTPException(404, "任务不存在")
            if data.get('status') == 'processing':
                raise HTTPException(202, "视频正在处理中，请稍后")
            elif data.get('status') == 'failed':
                raise HTTPException(500, "视频处理失败")
            else:
                raise HTTPException(404, "结果视频不存在")

        if request.method == "HEAD":
            return Response(headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(video_path.stat().st_size),
                "Accept-Ranges": "bytes"
            })

        return FileResponse(
            video_path,
            media_type="video/mp4",
            headers={"Content-Disposition": f"{'attachment' if download else 'inline'}; filename={task_id}_result.mp4"}
        )
    except HTTPException: raise
    except Exception as e:
        logger.error(f"获取视频失败 [{task_id}]: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/analyze/cancel/{task_id}")
async def cancel_task(task_id: str):
    if task_id in active_tasks:
        del active_tasks[task_id]
        save_task_status(task_id, "cancelled", -1, "用户已取消")
        logger.info(f"任务取消 [{task_id}]")
        return {"success": True}
    return {"success": False, "message": "任务不存在或已完成"}

@app.get("/health")
async def health_check():
    redis_status = "connected" if redis_client else "disconnected"
    return {
        "status": "ok",
        "version": "3.2.1-emergency-fix",
        "storage": {
            "redis": redis_status,
            "memory": len(memory_storage),
            "file_backup": len(list(status_backup_dir.glob("*.json")))
        },
        "image_enhancement": ENABLE_IMAGE_ENHANCE
    }

if __name__ == "__main__":
    import uvicorn
    Path("temp/uploads").mkdir(parents=True, exist_ok=True)
    logger.info("="*60)
    logger.info("启动紧急修复版服务 - 重点解决状态丢失问题")
    logger.info(f"图像增强: {ENABLE_IMAGE_ENHANCE}")
    logger.info(f"Redis: {'可用' if redis_client else '不可用'}")
    logger.info("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)