# 进行视频剪辑

from moviepy import VideoFileClip, concatenate_videoclips  # 去掉 .editor

# 1. 加载视频（改成你的路径）
input_path = r"D:\HUAWEI\Documents\毕业设计演示视频.mp4"   # 注意加 r，防止转义
output_path = r"D:\HUAWEI\Documents\演示视频.mp4"

# 2. 定义要剪掉的时间段（单位：秒）
cut_start = 46   # 从第10秒开始剪
cut_end = 300     # 剪到第20秒

# 3. 截取两段保留的部分
clip = VideoFileClip(input_path)
part1 = clip.subclipped(0, cut_start)           # 新版用 subclipped
part2 = clip.subclipped(cut_end, clip.duration)   # 新版用 subclipped

# 4. 拼接并输出
final_clip = concatenate_videoclips([part1, part2])
final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

# 5. 释放资源
clip.close()
final_clip.close()

print("剪辑完成！输出：", output_path)