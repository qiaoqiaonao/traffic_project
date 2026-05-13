import cv2
import numpy as np
from typing import List, Dict
from dataclasses import dataclass, field
import logging
from collections import deque, Counter

logger = logging.getLogger(__name__)


@dataclass
class Track:
    track_id: int
    bbox: List[float]
    score: float
    hits: int = 0
    time_since_update: int = 0
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0])
    color_hist: np.ndarray = None
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))
    status: str = "tentative"
    class_name: str = "car"
    class_history: deque = field(default_factory=lambda: deque(maxlen=15))

    def __post_init__(self):
        if self.color_hist is None:
            self.color_hist = np.zeros(512, dtype=np.float32)
        if len(self.trajectory) == 0:
            self.trajectory.append(tuple(self.bbox))
        if self.class_name and len(self.class_history) == 0:
            self.class_history.append(self.class_name)


class EnhancedDeepSORT:
    """
    增强版 DeepSORT
    固定摄像头：enable_cmc=False（默认），避免误判车辆运动为相机抖动
    无人机/手持：enable_cmc=True，补偿相机运动
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3,
                 appearance_weight=0.3, use_appearance=True,
                 enable_cmc=False):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self.use_appearance = use_appearance
        self.enable_cmc = enable_cmc

        self.tracks: List[Track] = []
        self.next_id = 1

        # CMC 相关（仅 enable_cmc=True 时使用）
        self.prev_gray = None

        logger.info(f"初始化增强版DeepSORT: use_appearance={use_appearance}, "
                    f"appearance_weight={appearance_weight}, enable_cmc={enable_cmc}")

    # ========== CMC：稀疏光流估计仿射矩阵（仅无人机/手持时启用）==========
    def _estimate_affine(self, frame: np.ndarray, detections: List[Dict] = None):
        if not self.enable_cmc:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        # 创建掩码，屏蔽检测框区域（避免运动车辆上的特征点干扰相机估计）
        mask = np.ones_like(self.prev_gray, dtype=np.uint8) * 255
        if detections:
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                margin = 30
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(mask.shape[1], x2 + margin)
                y2 = min(mask.shape[0], y2 + margin)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        pts0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200,
                                       qualityLevel=0.01, minDistance=30, mask=mask)
        if pts0 is None or len(pts0) < 10:
            self.prev_gray = gray
            return None

        pts1, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts0, None)
        valid = status.ravel() == 1
        pts0 = pts0[valid].reshape(-1, 2)
        pts1 = pts1[valid].reshape(-1, 2)

        if len(pts0) < 10:
            self.prev_gray = gray
            return None

        M, inliers = cv2.estimateAffinePartial2D(
            pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        # 剧烈晃动保护：位移/旋转过大则放弃 CMC
        if M is not None:
            tx, ty = abs(M[0, 2]), abs(M[1, 2])
            scale_x = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
            scale_y = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
            if tx > 30 or ty > 30 or not (0.9 < scale_x < 1.1) or not (0.9 < scale_y < 1.1):
                logger.debug(f"CMC 放弃: 位移({tx:.1f},{ty:.1f}) 或尺度异常")
                M = None

        self.prev_gray = gray
        return M

    def _warp_bbox(self, bbox: List[float], M: np.ndarray) -> List[float]:
        """对 bbox 做仿射变换（补偿相机运动）"""
        if M is None:
            return bbox
        pts = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
        pts = cv2.transform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
        return [float(pts[0, 0]), float(pts[0, 1]), float(pts[1, 0]), float(pts[1, 1])]

    # ========== 外观特征 ==========
    def _extract_color_feature(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(512, dtype=np.float32)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(512, dtype=np.float32)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 2],
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32)

    def _compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        return inter / (union + 1e-6) if union > 0 else 0

    def _compute_appearance_dist(self, hist1, hist2):
        if np.sum(hist1) == 0 or np.sum(hist2) == 0:
            return 1.0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    def _match_detections(self, detections, frame):
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        cost_matrix = np.zeros((len(detections), len(self.tracks)), dtype=np.float32)
        for i, det in enumerate(detections):
            det_hist = self._extract_color_feature(frame, det['bbox']) if self.use_appearance else None
            for j, track in enumerate(self.tracks):
                iou = self._compute_iou(det['bbox'], track.bbox)
                iou_dist = 1.0 - iou
                if self.use_appearance and det_hist is not None:
                    app_dist = self._compute_appearance_dist(det_hist, track.color_hist)
                    cost = (
                                       1 - self.appearance_weight) * iou_dist + self.appearance_weight * app_dist if iou > 0 else 1.0
                else:
                    cost = iou_dist
                cost_matrix[i, j] = cost

        matches = []
        matched_dets = set()
        matched_tracks = set()

        indices = np.argsort(cost_matrix.flatten())
        det_indices, track_indices = np.unravel_index(indices, cost_matrix.shape)

        for det_idx, track_idx in zip(det_indices, track_indices):
            if det_idx in matched_dets or track_idx in matched_tracks:
                continue
            iou = self._compute_iou(detections[det_idx]['bbox'], self.tracks[track_idx].bbox)
            cost = cost_matrix[det_idx, track_idx]
            if iou >= self.iou_threshold and cost < 0.8:
                matches.append((det_idx, track_idx))
                matched_dets.add(det_idx)
                matched_tracks.add(track_idx)

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_tracks]
        return matches, unmatched_dets, unmatched_tracks

    # ========== 核心 update ==========
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        # 1. CMC：估计并补偿相机运动（仅启用时）
        M = None
        if self.enable_cmc:
            M = self._estimate_affine(frame, detections)
            if M is not None:
                for track in self.tracks:
                    track.bbox = self._warp_bbox(track.bbox, M)

        # 2. 预测已有轨迹（匀速模型）
        for track in self.tracks:
            if len(track.trajectory) >= 2:
                last = track.trajectory[-1]
                prev = track.trajectory[-2]
                vx = (last[0] + last[2] - prev[0] - prev[2]) / 2
                vy = (last[1] + last[3] - prev[1] - prev[3]) / 2
                track.velocity = [vx, vy]
            track.bbox[0] += track.velocity[0]
            track.bbox[1] += track.velocity[1]
            track.bbox[2] += track.velocity[0]
            track.bbox[3] += track.velocity[1]
            track.time_since_update += 1

        # 3. 匹配
        matches, unmatched_dets, unmatched_tracks = self._match_detections(detections, frame)

        # 4. 更新匹配轨迹
        for det_idx, track_idx in matches:
            det = detections[det_idx]
            track = self.tracks[track_idx]
            track.bbox = det['bbox']
            track.score = det['score']

            # 类别投票平滑
            detected_class = det.get('class_name', 'car')
            track.class_history.append(detected_class)
            track.class_name = Counter(track.class_history).most_common(1)[0][0]

            track.time_since_update = 0
            track.hits += 1
            track.trajectory.append(tuple(det['bbox']))
            new_hist = self._extract_color_feature(frame, det['bbox'])
            track.color_hist = 0.9 * track.color_hist + 0.1 * new_hist
            if track.status == "tentative" and track.hits >= self.min_hits:
                track.status = "confirmed"

        # 5. 未匹配轨迹（标记丢失）
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            if track.time_since_update > self.max_age:
                track.status = "deleted"

        # 6. 未匹配检测（初始化新轨迹）
        # 仅在 CMC 启用且相机剧烈晃动时，抑制低置信度新轨迹
        is_camera_shaking = False
        if self.enable_cmc and M is not None:
            tx, ty = abs(M[0, 2]), abs(M[1, 2])
            is_camera_shaking = tx > 15 or ty > 15

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            # 晃动期间，仅对高置信度检测初始化新轨迹
            if is_camera_shaking and det['score'] < 0.7:
                continue

            hist = self._extract_color_feature(frame, det['bbox']) if self.use_appearance else np.zeros(512,
                                                                                                        dtype=np.float32)
            new_track = Track(
                track_id=self.next_id,
                bbox=det['bbox'],
                score=det['score'],
                hits=1,
                time_since_update=0,
                color_hist=hist,
                class_name=det.get('class_name', 'car')
            )
            new_track.trajectory.append(tuple(det['bbox']))
            self.tracks.append(new_track)
            self.next_id += 1

        # 7. 清理
        self.tracks = [t for t in self.tracks if t.status != "deleted"]

        # 8. 返回
        results = []
        for track in self.tracks:
            item = {
                'track_id': track.track_id,
                'bbox': track.bbox,
                'score': track.score,
                'class_name': track.class_name,
                'trajectory': list(track.trajectory),
                'status': track.status
            }
            if track.status == "tentative":
                item['is_tentative'] = True
            results.append(item)
        return results

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.prev_gray = None