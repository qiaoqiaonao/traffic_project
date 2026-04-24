import imageio.v3 as iio

video_path = "results/3f9eaedaf1f44cba_result.mp4"
meta = iio.immeta(video_path)
print(f"编码格式: {meta.get('codec', 'unknown')}")
print(f"FPS: {meta.get('fps', 'unknown')}")