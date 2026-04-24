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
    def __init__(self, model_path, use_onnx=None, conf_threshold=0.35, nms_threshold=0.5):
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
            self._init_onnx(model_path)
        else:
            raise ValueError("当前仅支持ONNX格式")

    def _init_onnx(self, model_path):
        """初始化ONNX Runtime"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

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

    def postprocess(self, outputs, orig_size, resized_img):
        """
        主格式：[class_id, score, x1, y1, x2, y2]（Paddle RT-DETR / UA-DETRAC 常见）。
        含最小面积过滤，抑制小片假阳性。
        """
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

        min_area = max(200.0, 0.00006 * float(orig_w * orig_h))
        detections = []

        if fmt == "class_score_box":
            for i in range(dets.shape[0]):
                cls_id = int(round(float(dets[i, 0])))
                score = float(dets[i, 1])
                if score < self.conf_threshold:
                    continue
                if cls_id < 0 or cls_id > 3:
                    continue
                x1, y1, x2, y2 = map(float, dets[i, 2:6])
                if x2 <= x1 or y2 <= y1:
                    continue
                x1o, y1o, x2o, y2o = self._map_box_to_original(x1, y1, x2, y2, orig_w, orig_h)
                area = max(0, x2o - x1o) * max(0, y2o - y1o)
                if area < min_area:
                    continue
                detections.append({
                    'bbox': [x1o, y1o, x2o, y2o],
                    'score': score,
                    'class_id': cls_id,
                    'class_name': self.class_names.get(cls_id, 'unknown')
                })
        else:
            for i in range(dets.shape[0]):
                score = float(dets[i, 1])
                if score < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = map(float, dets[i, 2:6])
                if x2 <= x1 or y2 <= y1:
                    continue
                cls_id = 0
                if dets.shape[1] > 6:
                    cls_id = int(round(float(dets[i, 6])))
                elif aux is not None and hasattr(aux, 'shape') and len(aux.shape) >= 2 and aux.shape[0] > i:
                    cls_id = int(np.argmax(aux[i]))
                cls_id = max(0, min(3, cls_id))
                x1o, y1o, x2o, y2o = self._map_box_to_original(x1, y1, x2, y2, orig_w, orig_h)
                area = max(0, x2o - x1o) * max(0, y2o - y1o)
                if area < min_area:
                    continue
                detections.append({
                    'bbox': [x1o, y1o, x2o, y2o],
                    'score': score,
                    'class_id': cls_id,
                    'class_name': self.class_names.get(cls_id, 'unknown')
                })

        detections.sort(key=lambda x: x['score'], reverse=True)
        detections = self._nms(detections)
        return detections

    def _nms(self, detections):
        """非极大值抑制（NMS）去除重叠框"""
        if len(detections) == 0:
            return []

        dets = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []

        while dets:
            current = dets.pop(0)
            keep.append(current)

            remaining = []
            for det in dets:
                iou = self._compute_iou(current['bbox'], det['bbox'])
                if iou < self.nms_threshold:
                    remaining.append(det)
            dets = remaining

        return keep

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

    def predict(self, image):
        """单张图片推理"""
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

        detections = self.postprocess(outputs, orig_size, resized_img)

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