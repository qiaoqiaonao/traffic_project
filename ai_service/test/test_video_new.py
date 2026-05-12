# ai_service/test_video.py
import cv2
import numpy as np
import time
from pathlib import Path
from collections import defaultdict
# 关键修改：从新的推理文件导入（支持4类和NMS）
from ai_service.test_inference_new import RTDETRPredictor


def process_video(
        video_path,
        model_path="ai_service/weights/rtdetr_detrac.onnx",
        output_path=None,
        conf_threshold=0.25,  # 建议0.25-0.3，平衡精度召回
        frame_skip=3,
        max_frames=None
):
    """
    处理视频文件，逐帧检测车辆（支持4类：car/bus/van/others）
    """

    # 初始化模型
    print(f"正在加载模型: {model_path}")
    predictor = RTDETRPredictor(
        model_path,
        use_onnx=True,
        conf_threshold=conf_threshold,
        nms_threshold=0.5  # NMS去重
    )
    print("模型加载完成")

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps}fps")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {total_frames / fps:.1f}秒")
    print(f"  检测策略: 每{frame_skip}帧检测一次")
    print(f"  检测类别: car, bus, van, others")

    # 准备输出视频
    writer = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"  将保存结果到: {output_path}")

    # 处理循环
    frame_count = 0
    detect_count = 0
    process_times = []
    last_detections = []  # 缓存上一帧结果用于跳帧显示

    # 新增：类别统计（用于最终报告）
    class_stats = defaultdict(int)

    print(f"\n开始处理...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 限制最大帧数
        if max_frames and frame_count >= max_frames:
            print(f"已达到最大帧数限制: {max_frames}")
            break

        current_time = frame_count / fps

        # 是否进行检测（跳帧逻辑）
        should_detect = (frame_count % frame_skip == 0)

        if should_detect:
            # 进行检测
            start = time.time()
            detections, infer_time = predictor.predict(frame)
            process_times.append(infer_time)

            last_detections = detections
            detect_count += len(detections)

            # 统计各类别
            for det in detections:
                class_stats[det['class_name']] += 1

            # 打印进度（每30帧或第一帧），显示详细类别分布
            if frame_count % 30 == 0 or frame_count == 0:
                avg_time = np.mean(process_times[-10:]) * 1000 if process_times else 0
                # 构建类别分布字符串
                class_dist = ", ".join([f"{k}:{v}" for k, v in class_stats.items()]) if class_stats else "无"
                print(f"  帧{frame_count:4d} ({current_time:5.1f}s): "
                      f"检测到{len(detections)}个目标 ({class_dist}), "
                      f"推理{infer_time * 1000:.1f}ms")
        else:
            # 使用上一帧结果（简单复制，实际可用跟踪算法插值）
            detections = last_detections

        # 可视化（画框）
        if writer or frame_count % 30 == 0:
            vis_frame = predictor.visualize(frame, detections, max_boxes=20)

            # 关键修改：overlay信息包含4类统计
            total_cars = sum(1 for d in detections if d['class_name'] == 'car')
            total_bus = sum(1 for d in detections if d['class_name'] == 'bus')
            total_van = sum(1 for d in detections if d['class_name'] == 'van')
            total_others = sum(1 for d in detections if d['class_name'] == 'others')

            info_text = f"Frame:{frame_count} | Car:{total_cars} Bus:{total_bus} Van:{total_van} Other:{total_others}"
            cv2.putText(vis_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 添加时间戳
            time_text = f"Time: {current_time:.1f}s | Skip: {frame_skip}"
            cv2.putText(vis_frame, time_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if writer:
                writer.write(vis_frame)

        frame_count += 1

    # 释放资源
    cap.release()
    if writer:
        writer.release()

    # 统计报告
    if process_times:
        avg_infer = np.mean(process_times) * 1000
        total_time = sum(process_times)
        print(f"\n{'=' * 50}")
        print(f"处理完成:")
        print(f"  处理帧数: {frame_count}")
        print(f"  实际检测帧数: {len(process_times)} (每{frame_skip}帧)")
        print(f"  总检测目标数: {detect_count}")
        print(f"  类别分布: {dict(class_stats)}")  # 显示4类统计
        print(f"  平均推理时间: {avg_infer:.1f}ms ({1000 / avg_infer:.1f} FPS)")
        print(f"  总检测耗时: {total_time:.1f}秒")
        print(f"  预估处理完整视频需: {(total_frames / frame_skip) * avg_infer / 1000:.1f}秒")
        if output_path:
            print(f"  结果视频已保存: {output_path}")

    return {
        "total_frames": frame_count,
        "processed_frames": len(process_times),
        "avg_infer_ms": avg_infer if process_times else 0,
        "total_detections": detect_count,
        "class_distribution": dict(class_stats),  # 新增：返回类别分布
        "fps": 1000 / avg_infer if process_times else 0
    }


if __name__ == "__main__":
    # 配置（根据你的实际路径修改）
    VIDEO_PATH = "dataset/videos/test_0000007.mp4"  # 改为你的视频路径
    OUTPUT_PATH = "output/result_video7.mp4"

    # 关键参数
    CONF_THRESHOLD = 0.25  # UA-DETRAC建议0.25，密集场景可调0.2
    FRAME_SKIP = 3  # CPU建议3，GPU可设1（逐帧检测）

    # 检查文件存在
    if not Path(VIDEO_PATH).exists():
        print(f"错误: 视频文件不存在: {VIDEO_PATH}")
        print("请确保路径正确，或使用示例:")
        print('  VIDEO_PATH = "dataset/videos/test_0000002.mp4"')
        exit(1)

    # 运行
    result = process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        conf_threshold=CONF_THRESHOLD,
        frame_skip=FRAME_SKIP,
        max_frames=None  # 设为100可快速测试前100帧
    )

    print(f"\n详细统计: {result}")