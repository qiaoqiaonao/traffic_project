"""
交通流量分析系统 - AI服务（优化版）
整合开题报告改进点：
1. 增强DeepSORT（外观特征）
2. 轨迹插值
3. 图像自适应增强
"""
# ai_service/main_optimized.py
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import threading
import cv2
import numpy as np
import json
import time
import base64
import redis
import logging
import math
import subprocess
import queue  # 新增：异步帧解码需要
from typing import List, Optional, Dict
from datetime import datetime

# 导入优化模块
from config import config
from enhanced_tracker import EnhancedDeepSORT
from trajectory_interpolator import TrajectoryInterpolator
from image_enhancement import ImageEnhancer
from detection_line import DetectionLine, parse_detection_line, default_vertical_line

class AsyncFrameReader:
    """
    异步帧预读取器：用独立线程解码视频帧，解耦 IO 和推理
    让 CPU 在等待解码的同时进行推理，提升整体吞吐
    """
    def __init__(self, cap: cv2.VideoCapture, maxsize: int = 5, frame_skip: int = 1):
        self.cap = cap
        self.frame_skip = frame_skip
        self.frame_queue = queue.Queue(maxsize=maxsize)
        self.stopped = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        self._frame_counter = 0

    def _read_loop(self):
        while not self.stopped:
            if self.frame_queue.full():
                time.sleep(0.001)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self.frame_queue.put((False, None))
                break
            self._frame_counter += 1
            if self.frame_skip <= 1 or self._frame_counter % self.frame_skip == 0:
                self.frame_queue.put((True, frame))

    def read(self):
        """阻塞读取一帧，返回 (ret, frame)"""
        return self.frame_queue.get()

    def stop(self):
        self.stopped = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass


class StaticObjectFilter:
    """
    轨迹静止过滤器：轻微过滤背景误检，但保留 confirmed 真车
    """
    def __init__(self, min_displacement=15, min_history=5, max_history=15):
        self.min_displacement = min_displacement
        self.min_history = min_history
        self.max_history = max_history
        self.history = {}
        self.lifetime_disp = {}

    def update(self, tracks):
        current_ids = set()
        filtered = []

        for t in tracks:
            tid = t['track_id']
            current_ids.add(tid)
            bbox = t['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if tid not in self.history:
                self.history[tid] = []
                self.lifetime_disp[tid] = 0.0

            if len(self.history[tid]) > 0:
                last_cx, last_cy = self.history[tid][-1]
                step_disp = math.hypot(cx - last_cx, cy - last_cy)
                self.lifetime_disp[tid] += step_disp

            self.history[tid].append((cx, cy))
            if len(self.history[tid]) > self.max_history:
                self.history[tid].pop(0)

            # 历史不足先保留
            if len(self.history[tid]) < self.min_history:
                filtered.append(t)
                continue

            # 关键改动：confirmed 真车即使静止也保留；只有 tentative 且从未动过的才过滤
            is_confirmed = t.get('status') == 'confirmed'
            if not is_confirmed and self.lifetime_disp.get(tid, 0) < self.min_displacement:
                continue  # 过滤掉从未动过的幽灵目标

            filtered.append(t)

        # 清理消失轨迹
        for tid in list(self.history.keys()):
            if tid not in current_ids:
                del self.history[tid]
                self.lifetime_disp.pop(tid, None)

        return filtered

    def reset(self):
        self.history.clear()
        self.lifetime_disp.clear()


# ============ 统一日志配置（服务入口唯一控制） ============
# 使用 config.LOG_DIR，确保路径一致；启动时自动清空旧日志
config.LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = config.LOG_DIR / "ai_service.log"

# 启动时清空历史日志
if log_file.exists():
    log_file.write_text("")

# 配置根logger，捕获所有子模块（enhanced_tracker、interpolator等）日志
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
# 避免重复添加（热重载/调试时可能重复执行）
for h in root_logger.handlers[:]:
    if isinstance(h, logging.FileHandler):
        root_logger.removeHandler(h)

file_handler = logging.FileHandler(
    str(log_file),
    encoding='utf-8',
    mode='a'  # 用 'a' 追加，但启动时已清空，等效于全新文件
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# 业务模块logger（自动传播到根logger）
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 如需在控制台也看ERROR级别（一般不需要），取消下面注释：
# console_handler = logging.StreamHandler(sys.stdout)
# console_handler.setLevel(logging.ERROR)
# logger.addHandler(console_handler)

# 确保目录存在
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(exist_ok=True)
config.LOG_DIR.mkdir(exist_ok=True)

app = FastAPI(title="交通流量分析系统 - 推理服务（优化版）", version="3.1.0")

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
active_tasks = {}
_storage_lock = threading.Lock()
_tasks_lock = threading.Lock()

# ============ 初始化核心组件 ============

_model_file = Path(config.MODEL_PATH)
if not _model_file.is_file():
    logger.error(
        "未找到 ONNX 模型文件: %s\n请将 rtdetr_detrac.onnx 放入目录: %s",
        _model_file.resolve(),
        (config.BASE_DIR / "weights").resolve(),
    )

# 初始化检测器（保持原有逻辑，从test_inference_new导入）
try:
    from test_inference_new import RTDETRPredictor

    detector = RTDETRPredictor(
        config.MODEL_PATH,
        use_onnx=True,
        conf_threshold=config.CONF_THRESHOLD,
        nms_threshold=config.NMS_THRESHOLD,
        num_threads=config.ONNX_NUM_THREADS
    )
    logger.info("✅ RT-DETR模型加载成功")
    model_loaded = True
except Exception as e:
    logger.error(f"❌ 模型加载失败: {e}")
    model_loaded = False

# 初始化增强组件
tracker = EnhancedDeepSORT(
    max_age=config.TRACKER_MAX_AGE,
    min_hits=config.TRACKER_MIN_HITS,
    iou_threshold=config.TRACKER_IOU_THRESH,
    appearance_weight=config.APPEARANCE_WEIGHT,
    use_appearance=config.ENABLE_APPEARANCE,
    enable_cmc=config.ENABLE_CMC
)

interpolator = TrajectoryInterpolator(max_gap=config.MAX_INTERP_GAP)
enhancer = ImageEnhancer(enable_enhancement=config.ENABLE_ENHANCEMENT)

# 如果模型加载失败，使用简单模拟器（测试用）
if not model_loaded:
    class SimpleDetector:
        def __init__(self):
            self.class_names = ['car', 'bus', 'van', 'others']
            import random
            self.random = random

        def predict(self, frame):
            time.sleep(0.05)  # 模拟延迟
            h, w = frame.shape[:2]
            dets = []
            for i in range(self.random.randint(1, 5)):
                x1 = self.random.randint(50, w - 200)
                y1 = self.random.randint(50, h - 200)
                x2 = x1 + self.random.randint(60, 150)
                y2 = y1 + self.random.randint(60, 120)
                dets.append({
                    'class_name': self.random.choice(self.class_names),
                    'score': self.random.uniform(0.6, 0.95),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
            return dets, 0.05


    detector = SimpleDetector()
    logger.warning("⚠️ 使用模拟检测器（测试模式）")

# ============ 虚拟检测线和流量统计（保持原有） ============

class TrafficCounter:
    def __init__(
            self,
            lines: List[DetectionLine],
            frame_width: int,
            frame_height: int,
            fps: float,
            meters_per_pixel: float,
            frame_skip: int = 3
    ):
        self.lines = lines
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = max(fps, 1e-3)
        self.meters_per_pixel = meters_per_pixel
        self.frame_skip = frame_skip
        self.counts = {line.name: {'in': 0, 'out': 0} for line in lines}
        self.vehicle_history = {}
        self.violations = []
        self.stationary_frames = {}
        self.track_classes = {}
        self.track_last_pos = {}          # track_id -> (frame_idx, cx, cy)
        self.track_speed_history = {}     # track_id -> [speed_mps, ...]  用于超速/违停
        self.track_velocity_history = {}  # track_id -> [(vx, vy), ...]   用于逆行
        self.speed_samples_mps: List[float] = []
        self._parking_fired = set()
        self._wrong_direction_fired = set()
        self._speeding_fired = set()

        # ========== 异常检测参数 ==========
        # 违停
        self.parking_duration_sec = 3.0
        self.parking_frame_threshold = max(10, int(self.fps * self.parking_duration_sec / self.frame_skip))
        self.parking_speed_threshold_mps = 1.0
        self.parking_min_movement_mps = 1.5

        # 逆行：不依赖 one_way，任何反向穿越都检测
        self.wrong_direction_min_speed_mps = 1.0

        # 超速：默认 60 km/h，可通过环境变量覆盖
        import os
        speed_limit_kmh = float(os.environ.get("SPEED_LIMIT_KMH", "60"))
        self.speed_limit_mps = speed_limit_kmh / 3.6
        self.speeding_min_history = 3
        self.speeding_min_track_frames = 9
        self.speeding_max_valid_mps = 120.0 / 3.6

        # 拥堵
        self.congestion_vehicle_threshold = 5
        self.congestion_speed_threshold_mps = 4.0
        self.congestion_distance_px = 300

    def update(self, tracks: List[Dict], frame_idx: int, timestamp: float):
        current_tracks = {}
        frame_diag = math.hypot(self.frame_width, self.frame_height)
        still_px = max(5.0, 0.002 * frame_diag)

        track_data = {}
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            class_name = track.get('class_name', 'car')

            if track_id not in self.track_classes:
                self.track_classes[track_id] = class_name

            current_tracks[track_id] = {
                'center': center,
                'bbox': bbox,
                'class': self.track_classes[track_id],
                'is_interpolated': track.get('is_interpolated', False)
            }

            if track.get('is_interpolated', False):
                continue

            # ===== 速度计算 =====
            velocity = (0.0, 0.0)
            speed_mps = 0.0
            if track_id in self.track_last_pos:
                pf, px, py = self.track_last_pos[track_id]
                dt = (frame_idx - pf) / self.fps
                if dt > 1e-6:
                    dx = center[0] - px
                    dy = center[1] - py
                    velocity = (dx / dt, dy / dt)
                    speed_mps = math.hypot(dx, dy) * self.meters_per_pixel / dt

                    # 只保留合理速度（0 ~ 100km/h）的样本
                    if 0 < speed_mps < self.speeding_max_valid_mps:
                        if track_id not in self.track_speed_history:
                            self.track_speed_history[track_id] = []
                        self.track_speed_history[track_id].append(speed_mps)
                        if len(self.track_speed_history[track_id]) > 5:
                            self.track_speed_history[track_id].pop(0)
                        self.speed_samples_mps.append(speed_mps)
                        if len(self.speed_samples_mps) > 10000:
                            del self.speed_samples_mps[:5000]  # 保留后一半

                        # 同时记录速度向量（用于逆行判断）
                        if track_id not in self.track_velocity_history:
                            self.track_velocity_history[track_id] = []
                        self.track_velocity_history[track_id].append(velocity)
                        if len(self.track_velocity_history[track_id]) > 10:
                            self.track_velocity_history[track_id].pop(0)

            self.track_last_pos[track_id] = (frame_idx, center[0], center[1])
            track_data[track_id] = {
                'center': center,
                'velocity': velocity,
                'speed_mps': speed_mps,
                'class': class_name,
                'bbox': bbox
            }

            # 违停检测
            self._check_parking(track_id, center, speed_mps, frame_idx, timestamp, still_px)

            # 超速检测
            self._check_speeding(track_id, center, frame_idx, timestamp)

        # 检测线穿越（流量统计 + 逆行）
        for track_id, data in track_data.items():
            if track_id in self.vehicle_history:
                last_pos = self.vehicle_history[track_id]['center']
                center = data['center']

                for line in self.lines:
                    line_px = (
                        line.x1 * self.frame_width,
                        line.y1 * self.frame_height,
                        line.x2 * self.frame_width,
                        line.y2 * self.frame_height
                    )

                    if self._crosses_line(last_pos, center, line_px):
                        direction = self._determine_direction(last_pos, center, line)
                        vehicle_type = self.track_classes[track_id]

                        if direction == 'in':
                            self.counts[line.name]['in'] += 1
                            self.counts[line.name].setdefault(f'{vehicle_type}_in', 0)
                            self.counts[line.name][f'{vehicle_type}_in'] += 1
                        else:
                            self.counts[line.name]['out'] += 1
                            self.counts[line.name].setdefault(f'{vehicle_type}_out', 0)
                            self.counts[line.name][f'{vehicle_type}_out'] += 1

                            # 逆行检测：对 one_way 线路直接检查；对普通线路也检查速度方向
                            self._check_wrong_direction(
                                track_id, center, line, line_px,
                                frame_idx, timestamp, vehicle_type
                            )

        # 拥堵检测
        self._check_congestion(track_data, frame_idx, timestamp)

        self.vehicle_history = current_tracks

        # 每 300 帧清理一次过期历史数据，防止内存膨胀
        if frame_idx % 300 == 0:
            self._gc_stale_tracks(current_tracks)

        return {
            'counts': self.counts,
            'active_vehicles': len(current_tracks),
            'total_violations': len(self.violations)
        }

    # ========== 违停 ==========
    def _check_parking(self, track_id, center, speed_mps, frame_idx, timestamp, still_px):
        if track_id not in self.stationary_frames:
            self.stationary_frames[track_id] = {
                'position': center,
                'frames': 0,
                'start_frame': frame_idx,
                'last_moving_frame': frame_idx if speed_mps > self.parking_speed_threshold_mps else None
            }
            return

        info = self.stationary_frames[track_id]
        dx = center[0] - info['position'][0]
        dy = center[1] - info['position'][1]
        distance = math.hypot(dx, dy)

        is_stationary = (speed_mps < self.parking_speed_threshold_mps) or (distance < still_px)

        if is_stationary:
            info['frames'] += 1
        else:
            info['position'] = center
            info['frames'] = 0
            info['start_frame'] = frame_idx
            info['last_moving_frame'] = frame_idx

        if (info['frames'] > self.parking_frame_threshold and
                track_id not in self._parking_fired):

            has_moved = False
            if track_id in self.track_speed_history and len(self.track_speed_history[track_id]) > 0:
                max_speed = max(self.track_speed_history[track_id])
                has_moved = max_speed > self.parking_min_movement_mps

            near_line = self._is_near_any_line(center, margin=100)

            if has_moved and not near_line:
                duration_sec = info['frames'] * self.frame_skip / self.fps
                self.violations.append({
                    'type': 'illegal_parking',
                    'track_id': track_id,
                    'vehicle_type': self.track_classes.get(track_id, 'car'),
                    'timestamp': timestamp,
                    'frame': frame_idx,
                    'location': [float(center[0]), float(center[1])],
                    'duration_sec': round(duration_sec, 1),
                    'reason': 'stationary_timeout'
                })
                self._parking_fired.add(track_id)

    # ========== 逆行 ==========
    def _check_wrong_direction(self, track_id, center, line, line_px, frame_idx, timestamp, vehicle_type):
        if track_id not in self.track_velocity_history:
            return
        if len(self.track_velocity_history[track_id]) < 2:
            return
        if track_id in self._wrong_direction_fired:
            return

        recent_v = self.track_velocity_history[track_id][-3:]
        avg_vx = sum(v[0] for v in recent_v) / len(recent_v)
        avg_vy = sum(v[1] for v in recent_v) / len(recent_v)
        avg_speed_mps = math.hypot(avg_vx, avg_vy) * self.meters_per_pixel

        if avg_speed_mps < self.wrong_direction_min_speed_mps:
            return

        road_dx = line_px[2] - line_px[0]
        road_dy = line_px[3] - line_px[1]
        dot = avg_vx * road_dx + avg_vy * road_dy
        if dot < 0:
            self.violations.append({
                'type': 'wrong_direction',
                'track_id': track_id,
                'vehicle_type': vehicle_type,
                'timestamp': timestamp,
                'frame': frame_idx,
                'location': [float(center[0]), float(center[1])],
                'line': line.name,
                'reason': 'direction_opposite',
                'speed_kmh': round(avg_speed_mps * 3.6, 1)
            })
            self._wrong_direction_fired.add(track_id)

    # ========== 超速（中位数滤波） ==========
    def _check_speeding(self, track_id, center, frame_idx, timestamp):
        if track_id in self._speeding_fired:
            return

        history = self.track_speed_history.get(track_id, [])
        if len(history) < self.speeding_min_history:
            return

        # 用 real-time dt 校验而不是 frame_skip * count，避免全帧/跳1帧时永远不触发
        if track_id in self.track_last_pos:
            first_frame = self.track_last_pos[track_id][0]  # 该 track 第一次记录的真实帧号
            tracked_seconds = (frame_idx - first_frame) / self.fps
            if tracked_seconds < 0.3:  # 至少持续 0.3 秒
                return

        sorted_hist = sorted(history)
        avg_speed = sorted_hist[len(sorted_hist) // 2]

        if avg_speed <= self.speed_limit_mps:
            return

        self.violations.append({
            'type': 'speeding',
            'track_id': track_id,
            'vehicle_type': self.track_classes.get(track_id, 'car'),
            'timestamp': timestamp,
            'frame': frame_idx,
            'location': [float(center[0]), float(center[1])],
            'speed_kmh': round(avg_speed * 3.6, 1),
            'speed_limit_kmh': round(self.speed_limit_mps * 3.6, 1)
        })
        self._speeding_fired.add(track_id)

    # ========== 拥堵 ==========
    def _check_congestion(self, track_data, frame_idx, timestamp):
        if not track_data:
            return

        for line in self.lines:
            line_cx = (line.x1 + line.x2) / 2 * self.frame_width
            line_cy = (line.y1 + line.y2) / 2 * self.frame_height

            nearby = []
            for tid, data in track_data.items():
                cx, cy = data['center']
                dist = math.hypot(cx - line_cx, cy - line_cy)
                if dist < self.congestion_distance_px:
                    nearby.append(data)

            if len(nearby) >= self.congestion_vehicle_threshold:
                avg_speed = sum(d['speed_mps'] for d in nearby) / len(nearby)
                if avg_speed < self.congestion_speed_threshold_mps:
                    already = any(
                        v.get('line') == line.name and v['type'] == 'congestion'
                        for v in self.violations
                    )
                    if not already:
                        self.violations.append({
                            'type': 'congestion',
                            'line': line.name,
                            'timestamp': timestamp,
                            'frame': frame_idx,
                            'vehicle_count': len(nearby),
                            'avg_speed_kmh': round(avg_speed * 3.6, 1)
                        })

    def _is_near_any_line(self, center, margin=100):
        cx, cy = center
        for line in self.lines:
            x1 = line.x1 * self.frame_width
            y1 = line.y1 * self.frame_height
            x2 = line.x2 * self.frame_width
            y2 = line.y2 * self.frame_height

            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                dist = math.hypot(cx - x1, cy - y1)
            else:
                t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / (dx * dx + dy * dy)))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = math.hypot(cx - proj_x, cy - proj_y)

            if dist < margin:
                return True
        return False

    def _gc_stale_tracks(self, current_tracks):
        """清理已消失 track 在各 history dict 中的残留数据"""
        active_ids = set(current_tracks.keys())
        for storage in (self.track_last_pos, self.track_speed_history,
                         self.track_velocity_history, self.stationary_frames):
            stale = [tid for tid in storage if tid not in active_ids]
            for tid in stale:
                del storage[tid]

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

    def speed_summary(self) -> Dict:
        if not self.speed_samples_mps:
            return {
                'avg_mps': 0.0, 'max_mps': 0.0, 'avg_kmh': 0.0, 'max_kmh': 0.0,
                'samples': 0, 'meters_per_pixel_assumed': self.meters_per_pixel
            }
        arr = np.array(self.speed_samples_mps, dtype=np.float32)
        avg = float(np.mean(arr))
        mx = float(np.max(arr))
        return {
            'avg_mps': round(avg, 3),
            'max_mps': round(mx, 3),
            'avg_kmh': round(avg * 3.6, 2),
            'max_kmh': round(mx * 3.6, 2),
            'samples': len(self.speed_samples_mps),
            'meters_per_pixel_assumed': self.meters_per_pixel,
        }


# ============ 可视化工具（优化显示轨迹） ============

class ResultVisualizer:
    COLORS = {
        'car': (0, 255, 0),  # 绿
        'bus': (255, 165, 0),  # 橙
        'van': (0, 0, 255),  # 红
        'others': (128, 128, 128),  # 灰
        'default': (255, 255, 0)  # 青
    }

    @staticmethod
    def draw_detection(frame: np.ndarray, track: Dict):
        bbox = track['bbox']
        track_id = track.get('track_id', 0)
        score = track.get('score', 0)
        class_name = track.get('class_name', 'unknown')
        is_interp = track.get('is_interpolated', False)

        # 颜色：confirmed=绿，interp=灰（tentative 不再特殊标记，和 confirmed 一样）
        if is_interp:
            color = (128, 128, 128)
        else:
            color = ResultVisualizer.COLORS.get(class_name, (0, 255, 0))

        x1, y1, x2, y2 = map(int, bbox)
        thickness = 1 if is_interp else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} ID:{track_id} {score:.2f}"
        if is_interp:
            label += " (interp)"

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame

    @staticmethod
    def draw_detection_line(frame: np.ndarray, line: DetectionLine, width: int, height: int):
        x1 = int(line.x1 * width)
        y1 = int(line.y1 * height)
        x2 = int(line.x2 * width)
        y2 = int(line.y2 * height)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(frame, line.name, (mid_x + 10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    @staticmethod
    def draw_stats(frame: np.ndarray, stats: Dict, counter: TrafficCounter, frame_idx: int, fps: float,
                   class_stats: Dict):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        y_offset = 40
        texts = [
            f"Frame: {frame_idx}",
            f"Active Vehicles: {stats['active_vehicles']}",
            f"Violations: {stats['total_violations']}",
            f"Lighting: {enhancer.get_lighting_info(frame)}",
            f"FPS: {fps:.1f}"
        ]

        # 添加各类别统计
        for cls, count in class_stats.items():
            texts.append(f"{cls}: {count}")

        for text in texts:
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        # 流量统计
        y_offset += 10
        for line_name, counts in stats['counts'].items():
            text = f"{line_name}: In={counts['in']} Out={counts['out']}"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            y_offset += 20

        return frame


visualizer = ResultVisualizer()


# ============ 任务状态管理（保持原有） ============

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
        with _storage_lock:
            memory_storage[task_id] = data


def get_task_status(task_id: str) -> Optional[Dict]:
    if redis_client:
        return redis_client.hgetall(f"task:{task_id}") or None
    with _storage_lock:
        return memory_storage.get(task_id)


# ============ 核心视频处理任务（优化版） ============

def process_video_task(task_id: str, video_path: Path, frame_skip: int = 3,
                       detection_lines: Optional[List[Dict]] = None,
                       meters_per_pixel: float = 0.05):
    """处理视频（整合开题报告所有优化点）"""
    cap = None
    out = None
    is_cancelled = False
    temp_output = None

    # 重置跟踪器状态（每视频独立）
    global tracker, interpolator
    tracker.reset()
    interpolator.reset()

    # ✅ 新增：初始化静止过滤器
    static_filter = StaticObjectFilter(min_displacement=40, min_history=5)


    try:
        logger.info(f"任务 {task_id}: 开始处理 {video_path}")
        update_task_status(task_id, "processing", 5, "初始化视频...")

        # 解析检测线（保持原有逻辑）
        lines_data = []
        if detection_lines and isinstance(detection_lines, list):
            lines_data = detection_lines
        elif detection_lines and isinstance(detection_lines, str):
            try:
                lines_data = json.loads(detection_lines)
            except:
                logger.warning("解析detection_lines失败")

        if not lines_data:
            lines_data = [{'name': 'main_line', 'x1': 0.5, 'y1': 0.0, 'x2': 0.5, 'y2': 1.0, 'direction': 'vertical'}]

        validated_lines = [x for x in (parse_detection_line(line) for line in lines_data) if x is not None]
        if not validated_lines:
            validated_lines = [default_vertical_line()]

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 输出路径
        final_output = config.RESULTS_DIR / f"{task_id}_result.mp4"
        temp_output = config.RESULTS_DIR / f"{task_id}_temp.mp4"

        if final_output.exists(): final_output.unlink()
        if temp_output.exists(): temp_output.unlink()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_output), fourcc, fps / frame_skip, (width, height))

        # ===== 实时HLS流输出 =====
        hls_dir = config.RESULTS_DIR / f"{task_id}_hls"
        hls_dir.mkdir(parents=True, exist_ok=True)
        for f in hls_dir.glob("*"): f.unlink()
        hls_proc = None
        hls_ffmpeg_ok = False
        try:
            # 检查FFmpeg可用性
            check = subprocess.run([config.FFMPEG_PATH, '-version'], capture_output=True, timeout=5)
            if check.returncode != 0:
                raise RuntimeError(f"FFmpeg不可用: {check.stderr.decode(errors='replace')}")
            hls_ffmpeg_ok = True
        except Exception as e:
            logger.error(f"HLS无法启动: FFmpeg检查失败 - {e}")

        if hls_ffmpeg_ok:
            try:
                hls_log = open(hls_dir / "ffmpeg.log", "w")
                hls_proc = subprocess.Popen([
                    config.FFMPEG_PATH, '-y',
                    '-f', 'image2pipe', '-vcodec', 'mjpeg',
                    '-use_wallclock_as_timestamps', '1',
                    '-i', 'pipe:0',
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
                    '-crf', '28', '-g', '10',
                    '-f', 'hls', '-hls_time', '3', '-hls_list_size', '8',
                    '-hls_flags', 'delete_segments+omit_endlist',
                    '-hls_segment_filename', str(hls_dir / 'seg_%03d.ts'),
                    str(hls_dir / 'stream.m3u8')
                ], stdin=subprocess.PIPE, stderr=hls_log)
                logger.info(f"HLS已启动 -> {hls_dir / 'stream.m3u8'}")
            except Exception as e:
                logger.error(f"HLS启动失败: {e}")
                hls_proc = None

        lines = validated_lines
        counter = TrafficCounter(lines, width, height, fps, meters_per_pixel, frame_skip)

        frame_results = []
        frame_count = 0
        processed_count = 0
        total_detections = 0
        last_frame_sent_time = 0.0
        last_progress_time = 0.0
        video_file_size = video_path.stat().st_size if video_path.exists() else 0
        class_stats = {'car': 0, 'bus': 0, 'van': 0, 'others': 0}

        update_task_status(task_id, "processing", 10, "开始分析...")

        # ===== 优化：启动异步帧解码器 =====
        reader_skip = 1 if frame_skip > 1 else 1
        frame_reader = AsyncFrameReader(cap, maxsize=5, frame_skip=reader_skip)
        logger.info(f"异步帧解码已启动 (buffer=5)")

        while True:
            with _tasks_lock:
                if task_id not in active_tasks:
                    is_cancelled = True
                    break

            if is_cancelled:
                break

            # ===== 优化：从异步读取器获取帧（非阻塞等待）=====
            ret, frame = frame_reader.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            timestamp = frame_count / fps

            # ===== 优化点1: 图像增强 =====
            process_frame = enhancer.enhance(frame)

            # ===== 优化点2: 自适应阈值（先计算，再传入推理，不修改全局状态） =====
            adaptive_conf = enhancer.get_adaptive_confidence(frame, config.CONF_THRESHOLD)

            # 检测
            try:
                detections, infer_time = detector.predict(process_frame, conf_threshold=adaptive_conf)

                # ✅ 检测级二次过滤：面积 + 长宽比 + 边缘
                h, w = frame.shape[:2]
                frame_area = h * w
                margin = min(w, h) * 0.08  # 边缘 8% 区域
                filtered_dets = []
                for d in detections:
                    bbox = d['bbox']
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    box_area = bw * bh
                    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

                    # 1. 面积：过滤巨型框（超过画面 8%）
                    if box_area > frame_area * 0.08:
                        continue
                    # 2. 长宽比：过滤极端比例
                    ratio = bw / max(bh, 1)
                    if ratio > 4.0 or ratio < 0.25:
                        continue
                    # 3. 边缘过滤：中心点太靠近边缘的，大概率是假阳性
                    if cx < margin or cx > w - margin or cy < margin or cy > h - margin:
                        continue

                    filtered_dets.append(d)
                detections = filtered_dets

                # ⚠️ 核心修复：tracker.update 只调用一次！
                tracks = tracker.update(detections, frame)

                # 跟踪后过滤：杀掉长期静止的背景轨迹
                tracks = static_filter.update(tracks)
            except Exception as e:
                logger.error(f"检测失败帧 {frame_count}: {e}")
                detections = []
                infer_time = 0
                tracks = []

            total_detections += len(detections)
            for det in detections:
                cls = det.get('class_name', 'unknown')
                if cls in class_stats:
                    class_stats[cls] += 1

            # === 修改：分离 confirmed / tentative，tentative 不进入插值和统计 ===
            confirmed_tracks = [t for t in tracks if t.get('status') == 'confirmed']
            tentative_tracks = [t for t in tracks if t.get('status') == 'tentative']
            confirmed_ids = {t['track_id'] for t in confirmed_tracks}

            # 只对 confirmed 轨迹插值，避免幽灵框被固化
            if config.ENABLE_INTERPOLATION and confirmed_tracks:
                confirmed_tracks = interpolator.update(confirmed_tracks, frame_count, confirmed_ids)

            # 合并用于可视化（tentative 显示灰色，但不参与流量统计）
            all_tracks = confirmed_tracks + tentative_tracks

            # 流量统计只对 confirmed，杜绝假阳性被计入
            stats = counter.update(confirmed_tracks, frame_count, timestamp)

            # 可视化：画出所有轨迹（confirmed + tentative），静止/运动车辆都标注
            vis_frame = frame.copy()
            for line in lines:
                visualizer.draw_detection_line(vis_frame, line, width, height)
            for track in all_tracks:  # ← 改这里：confirmed + tentative 都画
                visualizer.draw_detection(vis_frame, track)

            out.write(vis_frame)

            # HLS实时流：写标注帧到FFmpeg pipe
            if hls_proc and hls_proc.poll() is None:
                try:
                    _, jpeg = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    hls_proc.stdin.write(jpeg.tobytes())
                except Exception:
                    pass

            # 保存结果
            frame_data = {
                'frame': frame_count,
                'timestamp': round(timestamp, 2),
                'count': len(detections),
                'tracks': tracks,
                'traffic_stats': stats,
                'infer_ms': round(infer_time * 1000, 1)
            }
            frame_results.append(frame_data)
            if len(frame_results) > config.MAX_FRAME_RESULTS:
                frame_results = frame_results[-config.MAX_FRAME_RESULTS:]
            processed_count += 1

            # 更新进度：每10帧或超过2秒强制更新，保持前端进度条丝滑
            _now = time.time()
            if processed_count % 10 == 0 or (_now - last_progress_time > 2.0):
                last_progress_time = _now
                with _tasks_lock:
                    if task_id not in active_tasks:
                        is_cancelled = True
                if is_cancelled:
                    break

                # 进度：total_frames=0 时用文件字节位置估算
                if total_frames > 0:
                    progress = min(95, int(frame_count / total_frames * 100))
                elif video_file_size > 0:
                    progress = min(95, int(cap.get(cv2.CAP_PROP_POS_FRAMES) / max(total_frames, 1) * 100)) if total_frames > 0 else min(90, processed_count)
                else:
                    progress = min(90, processed_count)

                n_confirmed = len(confirmed_tracks)
                n_violations = len(counter.violations)
                frame_info = f"{frame_count}/{total_frames}" if total_frames > 0 else f"{frame_count}"
                update_task_status(task_id, "processing", progress,
                                   f"处理中... {frame_info}帧 | 轨迹:{n_confirmed} | 违规:{n_violations}")

        # 释放资源（先关HLS再关视频）
        if hls_proc and hls_proc.poll() is None:
            try: hls_proc.stdin.close()
            except: pass
            try: hls_proc.wait(timeout=5)
            except: hls_proc.kill()
        if out: out.release(); out = None
        if cap: cap.release(); cap = None

        if is_cancelled:
            logger.info(f"任务 {task_id}: 已取消")
            if temp_output.exists(): temp_output.unlink()
            return

        # FFmpeg转码（保持原有）
        if temp_output.exists() and temp_output.stat().st_size > 0:
            update_task_status(task_id, "processing", 96, "转码中...")
            ffmpeg_cmd = config.FFMPEG_PATH

            if Path(ffmpeg_cmd).exists():
                try:
                    cmd = [
                        ffmpeg_cmd, '-i', str(temp_output),
                        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                        '-preset', 'fast', '-crf', '23',
                        '-movflags', '+faststart', '-an', '-y',
                        str(final_output)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    if result.returncode == 0 and final_output.exists():
                        temp_output.unlink(missing_ok=True)
                    else:
                        temp_output.rename(final_output)
                except Exception as e:
                    logger.error(f"FFmpeg失败: {e}")
                    temp_output.rename(final_output)
            else:
                temp_output.rename(final_output)

        # 验证输出
        if not final_output.exists() or final_output.stat().st_size == 0:
            raise ValueError("视频生成失败")

        # 汇总结果
        avg_cars = round(total_detections / max(processed_count, 1), 4)
        final_result = {
            'video_info': {
                'total_frames': frame_count, 'fps': fps / frame_skip,
                'duration_sec': round(frame_count / fps, 1),
                'width': width, 'height': height
            },
            'statistics': {
                'total_detections': total_detections,
                'processed_frames': processed_count,
                'avg_cars_per_frame': avg_cars,
                'unique_vehicles': len(counter.track_classes),
                'traffic_counts': counter.counts,
                'class_distribution': class_stats,
                'speed_estimation': counter.speed_summary(),
                'violations': {
                    'total': len(counter.violations),
                    'details': counter.violations[:50]
                }
            },
            'output_files': {
                'result_video': str(final_output),
                'result_video_url': f"/api/analyze/video/{task_id}",
                'hls_url': f"/api/analyze/hls/{task_id}/stream.m3u8"
            },
            'frame_results': frame_results[:100]
        }

        # 违规诊断汇总
        v_summary = {}
        for v in counter.violations:
            v_summary[v['type']] = v_summary.get(v['type'], 0) + 1
        logger.info(
            f"任务 {task_id}: 完成 | 类别: {class_stats} | "
            f"违规({len(counter.violations)}): {v_summary} | "
            f"速度样本: {len(counter.speed_samples_mps)} | "
            f"参数: parking={counter.parking_frame_threshold}frames, "
            f"speed_limit={counter.speed_limit_mps*3.6:.0f}km/h, "
            f"congestion={counter.congestion_vehicle_threshold}veh"
        )

        update_task_status(
            task_id, "completed", 100,
            f"分析完成！检测到{len(counter.violations)}个违规事件"
            + (f": {v_summary}" if v_summary else f"（阈值: 超速>{counter.speed_limit_mps*3.6:.0f}km/h 违停>{counter.parking_duration_sec}s 拥堵>{counter.congestion_vehicle_threshold}车）"),
            result=final_result
        )

    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {str(e)}")
        update_task_status(task_id, "failed", -1, f"失败: {str(e)}", error=str(e))
        is_cancelled = True

    finally:
        # ===== 优化：确保异步读取器被关闭 =====
        if 'frame_reader' in locals() and frame_reader:
            frame_reader.stop()
        if out: out.release()
        if cap: cap.release()
        if temp_output and temp_output.exists():
            try:
                temp_output.unlink()
            except:
                pass
        with _tasks_lock:
            if task_id in active_tasks:
                del active_tasks[task_id]
        try:
            if video_path.exists(): video_path.unlink()
        except Exception as e:
            logger.warning(f"清理输入文件失败: {e}")


# ============ API端点（保持与原有完全一致，后端无需修改） ============

@app.post("/api/analyze/upload")
async def upload_video(
        background_tasks: BackgroundTasks,
        task_id: str = Form(...),
        file: UploadFile = File(...),
        frame_skip: int = Form(3),
        detection_lines: Optional[str] = Form(None),
        meters_per_pixel: float = Form(0.05)
):
    """上传视频（接口保持不变）"""
    try:
        allowed = {'.mp4', '.avi', '.mov', '.mkv'}
        ext = Path(file.filename or "unknown.mp4").suffix.lower()
        if ext not in allowed:
            raise HTTPException(400, f"不支持的格式: {ext}")

        if frame_skip < 1 or frame_skip > 10:
            raise HTTPException(400, "frame_skip必须在1-10之间")

        temp_dir = config.UPLOAD_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{task_id}{ext}"

        with _tasks_lock:
            active_tasks[task_id] = {
                "status": "processing", "start_time": time.time(),
                "video_path": str(video_path)
            }

        # 分块写入并检查文件大小限制
        max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        total_read = 0
        with open(video_path, "wb") as f:
            while chunk := file.file.read(8 * 1024 * 1024):  # 8MB chunks
                total_read += len(chunk)
                if total_read > max_bytes:
                    f.close()
                    video_path.unlink(missing_ok=True)
                    raise HTTPException(413, f"文件超过最大限制 {config.MAX_UPLOAD_SIZE_MB}MB")
                f.write(chunk)

        lines = None
        if detection_lines:
            try:
                lines = json.loads(detection_lines)
            except:
                raise HTTPException(400, "detection_lines格式错误")

        update_task_status(task_id, "pending", 0, "等待处理")
        background_tasks.add_task(process_video_task, task_id, video_path, frame_skip, lines, meters_per_pixel)

        return JSONResponse({
            "code": 200, "message": "上传成功，开始分析",
            "data": {
                "task_id": task_id, "status": "processing",
                "check_url": f"/api/analyze/result/{task_id}",
                "video_url": f"/api/analyze/video/{task_id}"
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传失败: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/analyze/result/{task_id}")
async def get_result(task_id: str):
    """查询结果（接口保持不变）"""
    try:
        task_data = get_task_status(task_id)
        if not task_data:
            return JSONResponse({"code": 404, "message": "任务不存在", "data": None})

        result = None
        if task_data.get('result'):
            try:
                result = json.loads(task_data['result']) if isinstance(task_data['result'], str) else task_data[
                    'result']
            except:
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
    """获取视频（接口保持不变）"""
    try:
        video_path = config.RESULTS_DIR / f"{task_id}_result.mp4"

        if not video_path.exists():
            task_data = get_task_status(task_id)
            if not task_data: raise HTTPException(404, "任务不存在")
            status = task_data.get('status')
            if status == 'processing':
                raise HTTPException(202, "处理中")
            elif status == 'failed':
                raise HTTPException(500, "处理失败")
            else:
                raise HTTPException(404, "视频不存在")

        if request.method == "HEAD":
            return Response(headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(video_path.stat().st_size),
                "Accept-Ranges": "bytes"
            })

        if download:
            return FileResponse(video_path, media_type="video/mp4", filename=f"{task_id}_result.mp4")
        else:
            return FileResponse(video_path, media_type="video/mp4",
                                headers={"Content-Disposition": f"inline; filename={task_id}_result.mp4"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/analyze/videos")
async def list_result_videos():
    """列出所有结果视频"""
    try:
        videos = []
        for video_file in config.RESULTS_DIR.glob("*_result.mp4"):
            task_id = video_file.stem.replace("_result", "")
            stat = video_file.stat()
            videos.append({
                "task_id": task_id, "filename": video_file.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "url": f"/api/analyze/video/{task_id}"
            })
        return JSONResponse({"code": 200, "data": {"total": len(videos),
                                                   "videos": sorted(videos, key=lambda x: x['created'], reverse=True)}})
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/analyze/video/{task_id}")
async def delete_result_video(task_id: str):
    """删除视频"""
    try:
        video_path = config.RESULTS_DIR / f"{task_id}_result.mp4"
        if video_path.exists():
            video_path.unlink()
            return JSONResponse({"code": 200, "message": "已删除"})
        raise HTTPException(404, "视频不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/analyze/frame")
async def analyze_single_frame(file: UploadFile = File(...)):
    """单帧检测（接口保持不变）"""
    try:
        allowed = {'.jpg', '.jpeg', '.png', '.bmp'}
        ext = Path(file.filename or "").suffix.lower()
        if ext not in allowed: raise HTTPException(400, f"不支持的格式: {ext}")

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: raise HTTPException(400, "无法读取图片")

        h, w = frame.shape[:2]

        # 增强预处理 + 自适应阈值（传入 predict，不修改全局状态）
        process_frame = enhancer.enhance(frame)
        adaptive_conf = enhancer.get_adaptive_confidence(frame, config.CONF_THRESHOLD)

        start = time.time()
        detections, _ = detector.predict(process_frame, conf_threshold=adaptive_conf)
        infer_time = time.time() - start

        # 绘制（保持与原有相同的颜色定义）
        colors = {'car': (0, 255, 0), 'bus': (255, 165, 0), 'van': (0, 0, 255), 'others': (128, 128, 128)}
        vis_frame = frame.copy()
        detection_results = []

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls = det['class_name']
            score = det['score']
            color = colors.get(cls, (128, 128, 128))

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            detection_results.append({
                'id': i + 1, 'class': cls, 'score': round(score, 3),
                'bbox': [x1, y1, x2, y2]
            })

        class_counts = {}
        for d in detection_results:
            cls = d['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        info_text = f"Total: {len(detections)} | " + " ".join([f"{k}:{v}" for k, v in class_counts.items()])
        cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', vis_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse({
            "code": 200,
            "data": {
                "count": len(detections),
                "detections": detection_results,
                "infer_time_ms": round(infer_time * 1000, 1),
                "image_width": w, "image_height": h,
                "marked_image": f"data:image/jpeg;base64,{img_base64}",
                "statistics": {"total": len(detections), "by_class": class_counts}
            }
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"单帧检测失败: {e}")
        raise HTTPException(500, str(e))


@app.api_route("/api/analyze/hls/{task_id}/{filename}", methods=["GET", "HEAD"])
async def serve_hls_file(task_id: str, filename: str):
    """实时HLS流文件服务"""
    path = config.RESULTS_DIR / f"{task_id}_hls" / filename
    if not path.exists():
        logger.warning(f"HLS 404: {path} (dir_exists={path.parent.exists()}, listing={list(path.parent.glob('*')) if path.parent.exists() else 'N/A'})")
        raise HTTPException(404, f"HLS文件尚未生成 (looking at {path})")
    if filename.endswith('.m3u8'):
        return FileResponse(path, media_type="application/vnd.apple.mpegurl")
    return FileResponse(path, media_type="video/mp2t")


@app.post("/api/analyze/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消任务"""
    with _tasks_lock:
        found = task_id in active_tasks
        if found:
            del active_tasks[task_id]
    if found:
        update_task_status(task_id, "cancelled", -1, "用户已取消")
        return {"success": True}
    return {"success": False, "message": "任务不存在"}

# 优雅关闭
@app.on_event("shutdown")
async def shutdown():
    logger.info("正在关闭服务...")
    # 取消所有进行中的任务
    with _tasks_lock:
        task_ids = list(active_tasks.keys())
        active_tasks.clear()
    for tid in task_ids:
        update_task_status(tid, "cancelled", -1, "服务关闭")
        logger.info(f"已取消任务: {tid}")
    if redis_client:
        try:
            redis_client.close()
        except:
            pass
    logger.info("服务已关闭")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": "rtdetr_detrac_4class" if model_loaded else "simulator",
        "enhancements": {
            "appearance_feature": config.ENABLE_APPEARANCE,
            "trajectory_interpolation": config.ENABLE_INTERPOLATION,
            "image_enhancement": config.ENABLE_ENHANCEMENT
        },
        "version": "3.1.0-optimized"
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 50)
    logger.info("启动优化版AI服务")
    logger.info(
        f"增强功能: 外观特征={config.ENABLE_APPEARANCE}, 插值={config.ENABLE_INTERPOLATION}, 图像增强={config.ENABLE_ENHANCEMENT}")
    logger.info("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)