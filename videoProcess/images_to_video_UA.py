import cv2
import os
import glob


def images_to_video(image_folder, output_video_path, fps=25):
    """
    将图片序列合成为视频，保持 videoProcess 原始 25fps 速率
    """
    # 获取所有 jpg 并按文件名排序（img00001, img00002...）
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    if not image_paths:
        print(f"❌ 未找到图片: {image_folder}")
        return

    print(f"✅ 找到 {len(image_paths)} 张图片，预计时长: {len(image_paths) / fps:.1f}秒")

    # 读取第一张获取尺寸
    first = cv2.imread(image_paths[0])
    height, width = first.shape[:2]

    # MP4 编码
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for i, path in enumerate(image_paths):
        frame = cv2.imread(path)
        if frame is None:
            continue
        # 尺寸统一（保险起见）
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)
        if (i + 1) % 100 == 0:
            print(f"   已处理 {i + 1}/{len(image_paths)}")

    out.release()
    print(f"✅ 完成: {output_video_path}")


# ==================== 配置 ====================
base_dir = r"D:\UA-DETRAC\archive (2)\DETRAC-Images\DETRAC-Images"
output_dir = r"D:\TestData\traffic\videos\UA-DETRAC"

folders = ["MVI_20011", "MVI_20012"]  # 按需添加更多

if __name__ == "__main__":
    for name in folders:
        inp = os.path.join(base_dir, name)
        out = os.path.join(output_dir, f"{name}.mp4")
        if os.path.exists(inp):
            images_to_video(inp, out, fps=25)
        else:
            print(f"❌ 跳过（不存在）: {inp}")