# ai_service/test_inference.py
import cv2
import numpy as np
import time
from pathlib import Path

try:
    import onnxruntime as ort

    USE_ONNX = True
    print("使用ONNX Runtime推理")
except ImportError:
    USE_ONNX = False
    print("使用Paddle Inference推理")


class RTDETRPredictor:
    def __init__(self, model_path, use_onnx=None, conf_threshold=0.3):
        self.input_size = 640
        self.conf_threshold = conf_threshold
        self.class_names = {0: 'car', 1: 'truck', 2: 'bus'}  # 模型输出0-based

        if use_onnx is None:
            use_onnx = USE_ONNX

        if use_onnx and not USE_ONNX:
            print("警告: ONNX不可用，回退到Paddle")
            use_onnx = False

        if use_onnx and Path(model_path).suffix == '.onnx':
            self._init_onnx(model_path)
        else:
            raise ValueError("当前仅支持ONNX格式，请确认模型路径以.onnx结尾")

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
        print(f"ONNX输入: {self.input_names}")
        print(f"ONNX输出: {self.output_names}")

    def preprocess(self, image):
        """
        预处理：直接resize到640x640（拉伸，不保持比例）
        RT-DETR的ONNX导出模型需要这种输入方式
        """
        orig_h, orig_w = image.shape[:2]

        # 直接resize到640x640（拉伸）
        resized = cv2.resize(image, (self.input_size, self.input_size))

        # BGR->RGB, 归一化, HWC->CHW
        input_tensor = resized[:, :, ::-1].astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)[None, ...]  # (1, 3, 640, 640)

        return input_tensor, (orig_h, orig_w), resized

    def postprocess(self, outputs, orig_size, resized_img):
        """
        后处理：解析输出并映射回原图尺寸
        输出格式: [batch_id, score, x1, y1, x2, y2]，坐标范围0-640
        """
        dets = outputs[0]  # (300, 6)

        orig_h, orig_w = orig_size
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        detections = []

        for i in range(dets.shape[0]):
            batch_idx = int(dets[i, 0])
            score = float(dets[i, 1])

            if score < self.conf_threshold:
                continue

            # 640x640空间内的坐标
            x1, y1, x2, y2 = dets[i, 2:6]

            # 检查坐标有效性
            if x2 <= x1 or y2 <= y1:
                continue

            # 映射回原图尺寸（简单拉伸映射）
            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y)
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)

            # 边界检查
            x1_orig = max(0, min(x1_orig, orig_w))
            y1_orig = max(0, min(y1_orig, orig_h))
            x2_orig = max(0, min(x2_orig, orig_w))
            y2_orig = max(0, min(y2_orig, orig_h))

            # 第6列是类别（0,1,2）
            cls_id = int(dets[i, 5]) if dets.shape[1] > 5 else 0
            # 或者根据你的模型输出，类别可能在其他位置
            # 如果第6列数值很大（如之前的1000+），说明不是类别，需要其他方式获取类别
            # 暂时先用0（car）作为默认，或根据训练时的类别映射调整

            # 如果类别值异常（>2），可能是未使用，设为0
            if cls_id > 2 or cls_id < 0:
                cls_id = 0

            detections.append({
                'bbox': [x1_orig, y1_orig, x2_orig, y2_orig],
                'score': score,
                'class_id': cls_id,
                'class_name': self.class_names.get(cls_id, 'unknown')
            })

        # 按分数排序
        detections.sort(key=lambda x: x['score'], reverse=True)

        return detections

    def predict(self, image):
        """
        单张图片推理
        返回: detections列表, 推理时间(秒)
        """
        # 预处理
        input_tensor, orig_size, resized_img = self.preprocess(image)

        # 准备ONNX输入
        # scale_factor设为1.0，因为我们直接resize到640x640，没有保持比例的缩放
        im_shape = np.array([[self.input_size, self.input_size]], dtype=np.float32)
        scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)  # 关键：设为1.0

        input_feed = {}
        for name in self.input_names:
            if 'image' in name or name == 'x':
                input_feed[name] = input_tensor
            elif 'im_shape' in name or 'shape' in name:
                input_feed[name] = im_shape
            elif 'scale' in name:
                input_feed[name] = scale_factor

        # 推理
        start = time.time()
        outputs = self.session.run(self.output_names, input_feed)
        inference_time = time.time() - start

        # 后处理
        detections = self.postprocess(outputs, orig_size, resized_img)

        return detections, inference_time

    def visualize(self, image, detections, save_path=None, max_boxes=50):
        """
        可视化检测结果
        max_boxes: 最多画多少个框（避免太多重叠）
        """
        result = image.copy()

        # 按分数排序后取前max_boxes个
        top_dets = detections[:max_boxes]

        for det in top_dets:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class_name']} {det['score']:.2f}"

            # 不同类别不同颜色
            colors = {
                'car': (0, 255, 0),  # 绿色
                'truck': (255, 0, 0),  # 蓝色
                'bus': (0, 0, 255),  # 红色
                'unknown': (128, 128, 128)  # 灰色
            }
            color = colors.get(det['class_name'], (0, 255, 0))

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, result)
            print(f"结果已保存: {save_path}")

        return result


def test_image(predictor, image_path):
    """测试单张图片"""
    print(f"\n{'=' * 50}")
    print(f"测试图片: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[错误] 无法读取图片: {image_path}")
        return None

    print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")

    # 推理
    detections, infer_time = predictor.predict(image)

    print(f"推理时间: {infer_time * 1000:.1f}ms")
    print(f"检测到 {len(detections)} 个目标 (阈值>{predictor.conf_threshold}):")

    # 显示前10个
    for i, det in enumerate(detections[:10]):
        print(f"  {i + 1}. {det['class_name']}: {det['score']:.3f} "
              f"box=[{det['bbox'][0]:.0f},{det['bbox'][1]:.0f},"
              f"{det['bbox'][2]:.0f},{det['bbox'][3]:.0f}]")

    # 可视化
    save_path = f"output/result_{Path(image_path).name}"
    predictor.visualize(image, detections, save_path, max_boxes=30)

    return detections


def test_video(predictor, video_path, max_frames=100, save_video=True):
    """
    测试视频
    save_video: 是否保存结果视频
    """
    print(f"\n{'=' * 50}")
    print(f"测试视频: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, {fps}fps, 总帧数:{total_frames}")

    # 准备输出视频
    if save_video:
        output_path = f"output/result_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"将保存结果视频到: {output_path}")

    frame_count = 0
    process_times = []

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 每3帧处理一次（加速）
        if frame_count % 3 == 0:
            detections, infer_time = predictor.predict(frame)
            process_times.append(infer_time)

            # 可视化
            vis_frame = predictor.visualize(frame, detections, max_boxes=20)

            if save_video:
                out.write(vis_frame)

            if frame_count % 30 == 0:
                avg_time = np.mean(process_times[-10:]) * 1000
                print(f"  帧{frame_count}: {len(detections)}辆车, "
                      f"{infer_time * 1000:.1f}ms, 平均{avg_time:.1f}ms")
        else:
            if save_video:
                out.write(frame)

        frame_count += 1

    cap.release()
    if save_video:
        out.release()

    if process_times:
        avg_time = np.mean(process_times) * 1000
        print(f"\n平均推理时间: {avg_time:.1f}ms ({1000 / avg_time:.1f} FPS)")
    print(f"处理了 {len(process_times)} 帧")


if __name__ == "__main__":
    # 配置
    model_path = "ai_service/weights/rtdetr_traffic.onnx"
    conf_threshold = 0.3  # 根据效果调整，小数据集可以设低一点如0.25

    # 初始化预测器
    print("正在加载模型...")
    predictor = RTDETRPredictor(model_path, use_onnx=True, conf_threshold=conf_threshold)
    print("模型加载完成")

    # 测试单张图片
    test_images = [
        "dataset/images/0000001_03499_d_0000006.jpg",
        # 添加更多图片...
    ]

    for img_path in test_images:
        if Path(img_path).exists():
            test_image(predictor, img_path)
        else:
            print(f"跳过不存在的图片: {img_path}")

    # 测试视频（如果有）
    # test_video_path = "dataset/videos/test.mp4"
    # if Path(test_video_path).exists():
    #     test_video(predictor, test_video_path, max_frames=300)

    print("\n全部测试完成！")