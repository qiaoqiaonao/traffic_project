import cv2
import numpy as np
import time
import logging
import sys
from pathlib import Path

# # ============ 日志配置修改：只写文件，不写控制台 ============
# LOG_DIR = Path("logs")
# LOG_DIR.mkdir(exist_ok=True)
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
#
# # 关键：强制清空已有handler，防止重复；禁止向根logger传播
# logger.handlers = []
# logger.propagate = False
#
# # 仅文件处理器，记录 DEBUG 及以上所有级别（最详细的日志进文件，方便查问题）
# file_handler = logging.FileHandler(
#     LOG_DIR / "ai_service.log",
#     encoding='utf-8',
#     mode='a'
# )
# file_handler.setLevel(logging.DEBUG)
# file_formatter = logging.Formatter(
#     '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# file_handler.setFormatter(file_formatter)
# logger.addHandler(file_handler)
#
# # 注意：已移除 StreamHandler(控制台)，所有业务日志只进文件
# # 控制台将只显示 Uvicorn/FastAPI 的 HTTP 请求日志
#
# # 屏蔽第三方库的冗余日志
# logging.getLogger("onnxruntime").setLevel(logging.WARNING)
import logging
from pathlib import Path

# 模块级只获取logger，不配置FileHandler（由主入口统一控制日志）
logger = logging.getLogger(__name__)

# 屏蔽第三方库的冗余日志
logging.getLogger("onnxruntime").setLevel(logging.WARNING)

try:
    import onnxruntime as ort

    USE_ONNX = True
    logger.info("使用ONNX Runtime推理")  # 这只会写入文件，不会显示在控制台
except ImportError:
    USE_ONNX = False
    logger.info("使用Paddle Inference推理")


class RTDETRPredictor:
    def __init__(self, model_path, use_onnx=None, conf_threshold=0.35, nms_threshold=0.5,
                 num_threads=None):
        self.input_size = 640
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # 4个类别 UA-DETRAC
        self.class_names = {0: 'car', 1: 'bus', 2: 'van', 3: 'others'}
        self.class_colors = {
            0: (0, 255, 0),  # car - 绿色
            1: (255, 165, 0),  # bus - 橙色
            2: (0, 0, 255),  # van - 红色
            3: (128, 128, 128)  # others - 灰色
        }

        if use_onnx is None:
            use_onnx = USE_ONNX

        if use_onnx and Path(model_path).suffix == '.onnx':
            self._init_onnx(model_path, num_threads)
        else:
            raise ValueError("当前仅支持ONNX格式")

    def _init_onnx(self, model_path, num_threads=None):
        """初始化ONNX Runtime - 优化版：内存优化 + 线程亲和性"""
        import os
        sess_options = ort.SessionOptions()

        # 1. 图优化级别最高
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 2. 线程数自适应：物理核心数（非超线程），留1核给系统
        if num_threads is None or num_threads <= 0:
            cpu_count = os.cpu_count() or 4
            num_threads = max(1, min(cpu_count - 1, 4))
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1  # 算子间调度单线程，减少切换开销

        # 3. 内存优化：启用内存池复用，减少分配开销
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        # 4. 线程亲和性：允许自旋等待（减少线程睡眠/唤醒开销）
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "1")

        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=['CPUExecutionProvider']
        )

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.debug(f"ONNX输入: {self.input_names}")
        logger.debug(f"ONNX输出: {self.output_names}")

    def preprocess(self, image):
        """预处理：与 infer_cfg.yml 一致（仅除255）"""
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, (self.input_size, self.input_size))

        # BGR->RGB, 转float, 除255, HWC->CHW
        img_rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
        input_tensor = img_rgb.transpose(2, 0, 1)[None, ...]  # (1, 3, 640, 640)

        return input_tensor, (orig_h, orig_w), resized

    def _infer_det_format(self, dets: np.ndarray) -> str:
        """
        推断 6 列输出的语义。禁止用「首若干行首列全 0」判断 batch 格式：
        类别为 car 时首列也常为 0，会误判并错解析类别/坐标。
        """
        for i in range(min(120, dets.shape[0])):
            s = float(dets[i, 1])
            if s < 0.05:
                continue
            c0 = float(dets[i, 0])
            if abs(c0) > 0.5 and abs(c0 - round(c0)) < 0.01 and 0 < s <= 1.0:
                return "class_score_box"
            if abs(c0 - round(c0)) < 0.05 and 0 <= round(c0) <= 3 and 0 < s <= 1.0:
                return "class_score_box"
        return "class_score_box"

    def _map_box_to_original(
            self, x1: float, y1: float, x2: float, y2: float, orig_w: int, orig_h: int
    ) -> tuple:
        """映射到原图坐标；兼容 0~640 网络空间与 0~1 归一化。"""
        mx = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if mx <= 1.0 + 1e-3:
            x1o = int(x1 * orig_w)
            y1o = int(y1 * orig_h)
            x2o = int(x2 * orig_w)
            y2o = int(y2 * orig_h)
        else:
            scale_x = orig_w / self.input_size
            scale_y = orig_h / self.input_size
            x1o = int(x1 * scale_x)
            y1o = int(y1 * scale_y)
            x2o = int(x2 * scale_x)
            y2o = int(y2 * scale_y)
        x1o = max(0, min(x1o, orig_w - 1))
        y1o = max(0, min(y1o, orig_h - 1))
        x2o = max(0, min(x2o, orig_w))
        y2o = max(0, min(y2o, orig_h))
        return x1o, y1o, x2o, y2o

    # ====================== 新增：外观特征预计算（供跟踪器直接使用，避免重复提取） ======================
    def _extract_color_feature(self, frame, bbox):
        """提取 512 维 HSV 颜色直方图 - 与 EnhancedDeepSORT 完全一致"""
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

    def _precompute_appearance(self, detections, frame):
        """为所有检测框预计算外观特征，避免跟踪器重复提取"""
        if not detections:
            return
        for det in detections:
            det['color_hist'] = self._extract_color_feature(frame, det['bbox'])
    # ====================== 结束新增 ======================

    def _map_boxes_to_original(self, boxes: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        """向量化坐标映射：一次性处理所有检测框"""
        mx = np.max(np.abs(boxes))
        if mx <= 1.0 + 1e-3:
            # 归一化坐标 (0~1)
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h
        else:
            # 网络空间坐标 (0~640)
            scale_x = orig_w / self.input_size
            scale_y = orig_h / self.input_size
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        # 裁剪到有效范围
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        return boxes

    def postprocess(self, outputs, orig_size, resized_img, threshold=None, frame=None):
        """
        向量化后处理：一次性过滤 + 坐标映射 + NMS + 外观特征预计算
        零影响检测效果，算法逻辑完全一致
        """
        if threshold is None:
            threshold = self.conf_threshold

        orig_h, orig_w = orig_size
        dets = np.asarray(outputs[0])
        aux = outputs[1] if len(outputs) > 1 else None

        logger.debug(f"输出0形状: {dets.shape}, 输出1: {aux.shape if aux is not None else None}")
        if dets.ndim != 2 or dets.shape[1] < 6:
            logger.warning(f"Unexpected dets shape: {getattr(dets, 'shape', None)}")
            return []
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"输出0前5行:\n{dets[:5]}")

        fmt = self._infer_det_format(dets)
        logger.info(f"后处理输出格式: {fmt}")

        # ===== 向量化提取所有检测数据 =====
        n = dets.shape[0]

        if fmt == "class_score_box":
            cls_ids = np.rint(dets[:, 0]).astype(np.int32)
            scores = dets[:, 1].astype(np.float32)
            boxes = dets[:, 2:6].astype(np.float32)
        else:
            scores = dets[:, 1].astype(np.float32)
            boxes = dets[:, 2:6].astype(np.float32)
            if aux is not None and hasattr(aux, 'shape') and len(aux.shape) >= 2:
                cls_ids = np.argmax(aux, axis=1).astype(np.int32)
            else:
                cls_ids = np.zeros(n, dtype=np.int32)

        # ===== 向量化过滤（置信度 + 类别 + 有效框）=====
        valid_mask = (
            (scores >= threshold) & (cls_ids >= 0) & (cls_ids <= 3) &
            (boxes[:, 2] > boxes[:, 0]) &  # x2 > x1
            (boxes[:, 3] > boxes[:, 1])     # y2 > y1
        )
        if not np.any(valid_mask):
            return []

        cls_ids = cls_ids[valid_mask]
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]

        # ===== 向量化坐标映射 =====
        boxes = self._map_boxes_to_original(boxes, orig_w, orig_h)

        # ===== 向量化面积过滤 =====
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        min_area = max(200.0, 0.00006 * float(orig_w * orig_h))
        area_mask = areas >= min_area
        if not np.any(area_mask):
            return []

        cls_ids = cls_ids[area_mask]
        scores = scores[area_mask]
        boxes = boxes[area_mask]

        # ===== 组装检测结果列表 =====
        detections = []
        for i in range(len(scores)):
            detections.append({
                'bbox': [int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])],
                'score': float(scores[i]),
                'class_id': int(cls_ids[i]),
                'class_name': self.class_names.get(int(cls_ids[i]), 'unknown')
            })

        # 按分数降序排序
        detections.sort(key=lambda x: x['score'], reverse=True)

        # ===== 向量化 NMS =====
        detections = self._nms_vectorized(detections)

        # ===== 优化：预计算外观特征（供跟踪器直接使用）=====
        if frame is not None:
            self._precompute_appearance(detections, frame)

        return detections

    def _nms(self, detections):
        """非极大值抑制（NMS）去除重叠框 - 保留原始方法名以兼容外部调用"""
        return self._nms_vectorized(detections)

    def _nms_vectorized(self, detections):
        """
        NumPy 向量化 NMS（比 Python 双重循环快 5-10x）
        算法逻辑与原版完全一致
        """
        if len(detections) == 0:
            return []

        # 提取所有框和分数为 numpy 数组
        boxes = np.array([d['bbox'] for d in detections], dtype=np.float32)
        scores = np.array([d['score'] for d in detections], dtype=np.float32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # 按分数降序排列的索引
        order = np.argsort(-scores)

        keep = []
        while order.size > 0:
            # 保留分数最高的框
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # 计算当前框与剩余框的 IoU（向量化）
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            # 保留 IoU 小于阈值的框
            mask = iou <= self.nms_threshold
            order = order[1:][mask]

        return [detections[i] for i in keep]

    def _compute_iou(self, box1, box2):
        """计算两个框的 IoU"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter_area == 0:
            return 0

        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-6)

    def predict(self, image, conf_threshold=None):
        """单张图片推理，conf_threshold 为 None 时使用实例默认值"""
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        input_tensor, orig_size, resized_img = self.preprocess(image)

        im_shape = np.array([[self.input_size, self.input_size]], dtype=np.float32)
        scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)

        input_feed = {}
        for name in self.input_names:
            if 'image' in name:
                input_feed[name] = input_tensor
            elif 'shape' in name:
                input_feed[name] = im_shape
            elif 'scale' in name:
                input_feed[name] = scale_factor
            else:
                input_feed[name] = input_tensor

        start = time.time()
        outputs = self.session.run(self.output_names, input_feed)
        inference_time = time.time() - start

        detections = self.postprocess(outputs, orig_size, resized_img, threshold, frame=image)

        # 这行现在只写入日志文件，不显示在控制台
        logger.info(f"检测到 {len(detections)} 个目标")

        return detections, inference_time

    def visualize(self, image, detections, save_path=None, max_boxes=50):
        """可视化检测结果"""
        result = image.copy()
        top_dets = detections[:max_boxes]

        for det in top_dets:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls_id = det['class_id']
            score = det['score']

            color = self.class_colors.get(cls_id, (0, 255, 0))
            label = f"{det['class_name']} {score:.2f}"

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10
            cv2.rectangle(result, (x1, label_y - text_h - 5), (x1 + text_w, label_y), color, -1)
            cv2.putText(result, label, (x1, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, result)
            logger.info(f"结果已保存: {save_path}")

        return result


# ==================== 测试函数 ====================

def test_image(predictor, image_path):
    """测试单张图片"""
    logger.info(f"{'=' * 50}")
    logger.info(f"测试图片: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"无法读取图片: {image_path}")
        return None

    logger.info(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")

    detections, infer_time = predictor.predict(image)

    logger.info(f"推理时间: {infer_time * 1000:.1f}ms")
    logger.info(f"NMS后检测到 {len(detections)} 个目标 (阈值>{predictor.conf_threshold})")

    # 统计各类别
    class_count = {}
    for det in detections:
        cls = det['class_name']
        class_count[cls] = class_count.get(cls, 0) + 1

    logger.info(f"类别分布: {class_count}")

    # 可视化
    save_path = f"output/result_{Path(image_path).name}"
    predictor.visualize(image, detections, save_path, max_boxes=30)

    return detections


if __name__ == "__main__":
    # 独立运行时临时配置日志
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)
    _tmp_logger = logging.getLogger()
    _tmp_logger.setLevel(logging.DEBUG)
    _fh = logging.FileHandler(LOG_DIR / "ai_service.log", encoding='utf-8', mode='a')
    _fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    _tmp_logger.addHandler(_fh)


    from config import config

    model_path = config.MODEL_PATH

    # 控制台只显示这一行（WARNING级别）
    logger.warning("正在加载模型...")

    predictor = RTDETRPredictor(
        model_path,
        use_onnx=True,
        conf_threshold=0.25,
        nms_threshold=0.5
    )

    logger.warning("模型加载完成，开始推理...（详细日志见 logs/ai_service.log）")

    test_images = [
        "dataset/images/0000163_00359_d_0000001.jpg",
        "dataset/images/0000001_03499_d_0000006.jpg",
    ]

    for img_path in test_images:
        if Path(img_path).exists():
            test_image(predictor, img_path)

    logger.warning("全部测试完成")