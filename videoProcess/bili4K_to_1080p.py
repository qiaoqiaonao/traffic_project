import cv2
import os
import glob


def resize_to_720p(input_path, output_path):
    """
    将视频缩放到 1280x720，保持原帧率，直接拉伸（适合交通检测）
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ 无法打开: {os.path.basename(input_path)}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_w, target_h = 1280, 720

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, target_h))
    if not out.isOpened():
        print(f"❌ 无法创建输出: {output_path}")
        cap.release()
        return False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 直接缩放到 1280x720（轻微变形，交通检测常用）
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        out.write(resized)

        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"   进度: {frame_idx}/{total_frames} ({frame_idx / total_frames * 100:.1f}%)")

    cap.release()
    out.release()

    # 验证输出
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"✅ 完成: {os.path.basename(output_path)} ({frame_idx}帧, {frame_idx / fps:.1f}秒)")
        return True
    else:
        print(f"❌ 输出失败: {output_path}")
        return False


# ==================== 批量处理 ====================

if __name__ == "__main__":
    # 源目录：你刚才剪辑好的 4K 片段
    src_dir = r"D:\TestData\traffic\videos\bilibili"

    # 目标目录：用户指定的存放位置
    dst_dir = r"D:\TestData\traffic\videos\bilibili1080p"

    # 查找所有 mp4（排除临时文件）
    videos = [f for f in glob.glob(os.path.join(src_dir, "*.mp4"))
              if not os.path.basename(f).startswith("__temp_")]

    if not videos:
        print("❌ 源目录没有找到 .mp4 文件")
    else:
        print(f"📁 找到 {len(videos)} 个视频，目标分辨率: 1280x720")
        success = 0
        for v in sorted(videos):
            name = os.path.basename(v)
            out = os.path.join(dst_dir, name)
            print(f"\n🔧 处理: {name}")
            if resize_to_720p(v, out):
                success += 1

        print(f"\n{'=' * 50}")
        print(f"🎉 全部完成！成功 {success}/{len(videos)} 个")
        print(f"📂 输出位置: {dst_dir}")
        print(f"{'=' * 50}")