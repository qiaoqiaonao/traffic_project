"""
配置文件 - 集中管理所有参数
模型路径基于本文件所在目录（ai_service/）解析，避免从项目根目录或 ai_service 目录启动时相对路径不一致。
"""
from pathlib import Path
from dataclasses import dataclass

# ai_service 包根目录（本文件所在目录）
_PKG_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    BASE_DIR: Path = _PKG_DIR
    # 权重文件放在 ai_service/weights/ 下（与启动工作目录无关）
    MODEL_PATH: str = str(_PKG_DIR / "weights" / "rtdetr_detrac.onnx")
    # 略提高默认阈值，减少空地/纹理误检；可按场景在环境变量或代码中调低
    CONF_THRESHOLD: float = 0.5
    NMS_THRESHOLD: float = 0.5
    INPUT_SIZE: int = 640

    # 跟踪器配置（开题报告优化点：减少ID切换）
    TRACKER_MAX_AGE: int = 3  # 最大丢失帧数
    TRACKER_MIN_HITS: int = 2  # 确认所需最少帧数
    TRACKER_IOU_THRESH: float = 0.3  # IOU阈值
    APPEARANCE_WEIGHT: float = 0.3  # 外观特征权重（0-1）
    ENABLE_APPEARANCE: bool = True  # 是否启用外观特征

    # ========== 方案2：相机运动补偿（CMC）总开关 ==========
    ENABLE_CMC: bool = False

    # 轨迹插值配置（开题报告优化点：遮挡恢复）
    ENABLE_INTERPOLATION: bool = True
    MAX_INTERP_GAP: int = 2  # 最大插值帧数（5帧约0.17秒）

    # 白天/高俯拍场景下 CLAHE+模糊易引入伪纹理导致误检，默认关闭；夜间弱光可改为 True
    ENABLE_ENHANCEMENT: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    NIGHT_BRIGHTNESS_THRESH: int = 50  # 夜间阈值
    DAY_BRIGHTNESS_THRESH: int = 200  # 白天阈值

    # 性能配置
    FRAME_SKIP: int = 3
    MAX_BOXES: int = 50

    # 车速估算：像素→米的近似标定（需按摄像头/场景标定，论文中可说明为经验参数）
    METERS_PER_PIXEL: float = 0.05

    # 路径配置（均相对 ai_service 目录，避免找不到 uploads/results）
    UPLOAD_DIR: Path = _PKG_DIR / "temp" / "uploads"
    RESULTS_DIR: Path = _PKG_DIR / "results"
    LOG_DIR: Path = _PKG_DIR / "logs"

config = Config()
