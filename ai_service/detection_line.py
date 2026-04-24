"""
检测线配置（单一来源，避免各入口 dataclass 不一致导致 one_way 等字段报错）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DetectionLine:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    direction: str
    one_way: bool = False


def parse_detection_line(line: Any) -> Optional[DetectionLine]:
    """从字典构建检测线，忽略未知字段，避免 **dict 与旧版 dataclass 不兼容。"""
    if not isinstance(line, dict):
        return None
    req = ("name", "x1", "y1", "x2", "y2", "direction")
    if not all(k in line for k in req):
        return None
    return DetectionLine(
        name=str(line["name"]),
        x1=float(line["x1"]),
        y1=float(line["y1"]),
        x2=float(line["x2"]),
        y2=float(line["y2"]),
        direction=str(line["direction"]),
        one_way=bool(line.get("one_way", False)),
    )


def default_vertical_line() -> DetectionLine:
    return DetectionLine("main_line", 0.5, 0.0, 0.5, 1.0, "vertical", False)
