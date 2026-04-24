# ai_service/test_video.py
import cv2
import numpy as np
import time
from pathlib import Path
from test_inference import RTDETRPredictor  # 导入你之前验证好的类


def process_video(
        video_path,
        model_path="ai_service/weights/rtdetr_detrac.onnx",
        output_path=None,
        conf_threshold=0.3,
        frame_skip=3,  # 每3帧检测一次（CPU优化）
        max_frames=None
):
    """
    处理视频文件，逐帧检测车辆

    Args:
        video_path: 输入视频路径
        model_path: ONNX模型路径
        output_path: 输出视频路径（None则不保存）
        conf_threshold: 置信度阈值
        frame_skip: 跳帧数（每N帧检测一次，中间帧用缓存）
        max_frames: 最大处理帧数（None则处理全部）
    """

    # 初始化模型
    print(f"正在加载模型: {model_path}")
    predictor = RTDETRPredictor(model_path, use_onnx=True, conf_threshold=conf_threshold)
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

    print(f"\n开始处理...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 限制最大帧数
        if max_frames and frame_count >= max_frames:
            print(f"已达到最大帧数限制: {max_frames}")
            break

        current_time = frame_count / fps  # 当前时间（秒）

        # 是否进行检测（跳帧逻辑）
        should_detect = (frame_count % frame_skip == 0)

        if should_detect:
            # 进行检测
            start = time.time()
            detections, infer_time = predictor.predict(frame)
            process_times.append(infer_time)

            last_detections = detections
            detect_count += len(detections)

            # 打印进度（每30帧或第一帧）
            if frame_count % 30 == 0 or frame_count == 0:
                avg_time = np.mean(process_times[-10:]) * 1000 if process_times else 0
                print(f"  帧{frame_count:4d} ({current_time:5.1f}s): "
                      f"检测到{len(detections)}辆车, "
                      f"推理{infer_time * 1000:.1f}ms, "
                      f"平均{avg_time:.1f}ms")
        else:
            # 使用上一帧结果（简单复制，实际可用跟踪算法插值）
            detections = last_detections

        # 可视化（画框）
        if writer or frame_count % 30 == 0:  # 保存视频或每30帧显示
            vis_frame = predictor.visualize(frame, detections, max_boxes=20)

            # 添加信息 overlay
            info_text = f"Frame: {frame_count} | Cars: {len(detections)} | Time: {current_time:.1f}s"
            cv2.putText(vis_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
        print(f"  总车辆出现次数: {detect_count}")
        print(f"  平均推理时间: {avg_infer:.1f}ms ({1000 / avg_infer:.1f} FPS)")
        print(f"  总检测耗时: {total_time:.1f}秒")
        if output_path:
            print(f"  结果视频已保存: {output_path}")

    return {
        "total_frames": frame_count,
        "processed_frames": len(process_times),
        "avg_infer_ms": avg_infer if process_times else 0,
        "total_detections": detect_count
    }


if __name__ == "__main__":
    # 配置
    VIDEO_PATH = "dataset/videos/test_0000002.mp4"  # 你生成的视频路径
    OUTPUT_PATH = "output/result_video2.mp4"  # 结果保存路径
    CONF_THRESHOLD = 0.3  # 阈值
    FRAME_SKIP = 3  # 每3帧检测一次（CPU建议3-5）

    # 运行
    result = process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        conf_threshold=CONF_THRESHOLD,
        frame_skip=FRAME_SKIP,
        max_frames=None  # 设为100可只测前100帧
    )

    print(f"\n统计: {result}")