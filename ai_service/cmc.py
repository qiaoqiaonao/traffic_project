import cv2
import numpy as np

class CameraMotionCompensator:
    """
    基于光流的相机运动补偿器 (CMC)
    估计相邻帧之间的 2D 仿射变换矩阵 M (prev -> cur)
    """
    def __init__(self, max_corners=300, quality_level=0.01, min_distance=30):
        self.prev_gray = None
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance

    def update(self, frame):
        """
        传入当前帧 BGR，返回 2x3 仿射变换矩阵 M
        M 满足: p_cur = M * p_prev  (齐次坐标)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return np.eye(2, 3, dtype=np.float32)

        # 1. 角点检测
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=7,
            useHarrisDetector=False
        )
        if prev_pts is None or len(prev_pts) < 6:
            self.prev_gray = gray
            return np.eye(2, 3, dtype=np.float32)

        # 2. 光流跟踪
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if curr_pts is None:
            self.prev_gray = gray
            return np.eye(2, 3, dtype=np.float32)

        # 3. 筛选有效点
        prev_pts = prev_pts[status.flatten() == 1]
        curr_pts = curr_pts[status.flatten() == 1]
        if len(prev_pts) < 6:
            self.prev_gray = gray
            return np.eye(2, 3, dtype=np.float32)

        # 4. RANSAC 估计仿射变换 (旋转 + 均匀缩放 + 平移)
        M, inliers = cv2.estimateAffinePartial2D(
            prev_pts, curr_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        if M is None or (inliers is not None and np.sum(inliers) < 6):
            M = np.eye(2, 3, dtype=np.float32)

        self.prev_gray = gray
        return M.astype(np.float32)


def apply_cmc_to_tracker(tracker, M):
    """
    对 DeepSORT Tracker 实例内部的 tracks 进行相机运动补偿。
    请在 tracker.predict() 之后、tracker.update() 之前调用。

    tracker: 你的 DeepSORT 内部 Tracker 实例 (需有 .tracks 属性)
    M: 2x3 仿射变换矩阵 (prev -> cur)
    """
    if M is None or not hasattr(tracker, 'tracks'):
        return

    for track in tracker.tracks:
        # 只处理已确认或上一帧刚更新的轨迹
        if not track.is_confirmed() and track.time_since_update > 0:
            continue

        # track.mean: [x, y, a, h, vx, vy, va, vh]
        # 1. 补偿中心点位置
        x, y = track.mean[0], track.mean[1]
        x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        track.mean[0], track.mean[1] = x_new, y_new

        # 2. 补偿尺度 (相机拉近/拉远/抖动)
        sx = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        sy = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
        scale = (sx + sy) / 2.0
        if scale > 0:
            track.mean[3] *= scale          # h
            track.mean[4:8] *= scale        # vx, vy, va, vh (速度也按像素尺度缩放)

        # 3. 简单补偿协方差 (位置/尺度不确定性)
        if hasattr(track, 'covariance'):
            track.covariance[:2, :2] *= (scale ** 2)
            track.covariance[3, 3] *= (scale ** 2)