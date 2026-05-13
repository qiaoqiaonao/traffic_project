import cv2
import os

def split_video(input_path, output_dir, segment_seconds=30):
    """
    将长视频按固定时长切割成多个片段
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {input_path}")
        return

    # 读取视频基础信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0 or total_frames == 0:
        print("❌ 视频信息读取失败")
        return

    # 每段多少帧（30秒 × 帧率）
    frames_per_segment = int(fps * segment_seconds)
    total_segments = (total_frames + frames_per_segment - 1) // frames_per_segment

    print(f"🎬 视频信息: {width}x{height}, {fps:.2f}fps, 总帧数: {total_frames}, 总时长: {total_frames/fps:.1f}s")
    print(f"✂️ 每段约 {segment_seconds}s ({frames_per_segment} 帧)，预计生成 {total_segments} 个片段")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 基于原视频文件名命名片段
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    segment_idx = 1      # 片段序号
    frame_idx = 0      # 当前读到第几帧
    out = None         # 当前写入器

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 每到新一段的开头，创建新的视频文件
        if frame_idx % frames_per_segment == 0:
            if out is not None:
                out.release()

            output_path = os.path.join(output_dir, f"{base_name}_{segment_idx:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"   写入片段 {segment_idx:03d}: {output_path}")
            segment_idx += 1

        out.write(frame)
        frame_idx += 1

        # 每 200 帧打印一次进度
        if frame_idx % 200 == 0:
            print(f"      进度: {frame_idx}/{total_frames} 帧 ({frame_idx/total_frames*100:.1f}%)")

    # 收尾
    if out is not None:
        out.release()
    cap.release()

    print(f"\n✅ 全部完成！共生成 {segment_idx - 1} 个片段，保存在:\n   {output_dir}")


# ==================== 配置区（只改这里） ====================

if __name__ == "__main__":
    # 你的长视频路径（根据你截图的位置，如果视频在别处请修改）
    input_video = r"D:\aDrive download\video\traffic.mp4"

    # 输出目录
    output_dir = r"D:\TestData\traffic\videos\bilibili"

    # 每个片段时长（秒）
    segment_seconds = 30

    split_video(input_video, output_dir, segment_seconds)