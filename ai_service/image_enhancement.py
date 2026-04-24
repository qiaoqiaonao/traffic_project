"""
图像预处理增强 - 提升复杂环境（夜间、逆光）下的检测效果
开题报告优化点：复杂环境下系统泛化能力
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    自适应图像增强：
    1. CLAHE自适应直方图均衡（改善逆光/夜间对比度）
    2. 动态检测阈值调整（夜间降低阈值提高召回）
    """

    def __init__(self, enable_enhancement=True, clahe_limit=2.0):
        self.enable = enable_enhancement
        self.clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        图像增强处理
        """
        if not self.enable:
            return frame

        # 1. 自适应直方图均衡（Lab颜色空间）
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # 2. 轻微去噪（保持边缘的快速NLM）
        # 注意：为了速度，这里使用高斯模糊代替NLM，实际部署可升级
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0.5)

        return denoised

    def get_adaptive_confidence(self, frame: np.ndarray, base_conf: float) -> float:
        """
        仅在极暗场景略降阈值提高召回；正常/强光场景保持 base_conf，避免误检增多。
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 50:
            return max(0.15, base_conf * 0.85)
        return base_conf

    def get_lighting_info(self, frame: np.ndarray) -> str:
        """返回当前光照情况，用于日志"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 50:
            return "night"
        elif mean_brightness > 200:
            return "strong_light"
        else:
            return "normal"