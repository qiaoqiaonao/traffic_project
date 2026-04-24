"""
轨迹插值器 - 处理遮挡导致的轨迹断裂
开题报告优化点：轨迹插值与重识别机制
"""
import numpy as np
from typing import List, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TrajectoryInterpolator:
    """
    轨迹插值器：当目标暂时遮挡（5帧以内）时，使用线性插值补全轨迹
    """

    def __init__(self, max_gap: int = 5):
        self.max_gap = max_gap
        # 历史记录：track_id -> [{'frame': int, 'bbox': [...]}, ...]
        self.history = defaultdict(list)

    def update(self, tracks: List[Dict], frame_idx: int, confirmed_track_ids: set = None) -> List[Dict]:
        current_ids = {t['track_id'] for t in tracks}

        # 检查哪些轨迹丢失了（可能遮挡）
        for track_id, hist in list(self.history.items()):
            if track_id in current_ids:
                continue

            # 只给 confirmed 轨迹插值，避免幽灵轨迹被固化
            if confirmed_track_ids is not None and track_id not in confirmed_track_ids:
                continue

            last_record = hist[-1]
            gap = frame_idx - last_record['frame']

            if gap > self.max_gap:
                continue

            if len(hist) < 2:
                continue

            prev_record = hist[-2]
            last_bbox = last_record['bbox']
            prev_bbox = prev_record['bbox']

            frame_diff = last_record['frame'] - prev_record['frame']
            if frame_diff == 0:
                continue

            vx = (last_bbox[0] - prev_bbox[0]) / frame_diff
            vy = (last_bbox[1] - prev_bbox[1]) / frame_diff

            interp_bbox = [
                last_bbox[0] + vx * gap,
                last_bbox[1] + vy * gap,
                last_bbox[2] + vx * gap,
                last_bbox[3] + vy * gap
            ]

            interp_track = {
                'track_id': track_id,
                'bbox': interp_bbox,
                'score': 0.5,
                'class_name': last_record.get('class_name', 'car'),
                'is_interpolated': True,
                'interpolation_gap': gap,
                'status': 'confirmed'
            }

            tracks.append(interp_track)
            logger.debug(f"轨迹 {track_id} 帧 {frame_idx} 已插值（丢失 {gap} 帧）")

        # 更新历史记录（只记录真实检测，不记录插值）
        for track in tracks:
            track_id = track['track_id']
            if not track.get('is_interpolated', False):
                self.history[track_id].append({
                    'frame': frame_idx,
                    'bbox': track['bbox'],
                    'class_name': track.get('class_name', 'car')
                })
                if len(self.history[track_id]) > self.max_gap + 3:
                    self.history[track_id].pop(0)

        # 清理长期未见的轨迹历史
        self.history = defaultdict(list, {
            k: v for k, v in self.history.items()
            if frame_idx - v[-1]['frame'] <= self.max_gap * 2
        })

        return tracks

    def reset(self):
        """重置历史"""
        self.history.clear()