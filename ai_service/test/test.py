import cv2
import numpy as np
from pathlib import Path
import onnxruntime as ort

# 加载模型
model_path = "ai_service/weights/rtdetr_traffic.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_names = [inp.name for inp in session.get_inputs()]

# 加载图片
image_path = "dataset/images/0000026_00500_d_0000025.jpg"
orig_image = cv2.imread(image_path)
h_img, w_img = orig_image.shape[:2]
print(f"原图尺寸: {w_img}x{h_img}")

# 直接resize到640x640（不保持比例，拉伸）
input_size = 640
resized_640 = cv2.resize(orig_image, (input_size, input_size))

# 预处理
input_tensor = resized_640[:, :, ::-1].astype(np.float32) / 255.0
input_tensor = input_tensor.transpose(2, 0, 1)[None, ...]

# 推理（简化，不用scale_factor）
im_shape = np.array([[input_size, input_size]], dtype=np.float32)
# scale_factor设为1，表示无缩放
scale_factor = np.array([[1.0, 1.0]], dtype=np.float32)

input_feed = {}
for name in input_names:
    if 'image' in name:
        input_feed[name] = input_tensor
    elif 'im_shape' in name:
        input_feed[name] = im_shape
    elif 'scale' in name:
        input_feed[name] = scale_factor

outputs = session.run(None, input_feed)
dets = outputs[0]

print(f"输出形状: {dets.shape}")
print(f"前3个:\n{dets[:3]}")

# 在640x640图像上画框
debug_img_640 = resized_640.copy()
count = 0

for i in range(300):
    score = dets[i, 1]
    if score < 0.3:
        continue

    x1, y1, x2, y2 = map(int, dets[i, 2:6])

    # 检查是否在640范围内
    if x1 < 0 or y1 < 0 or x2 > 640 or y2 > 640 or x2 <= x1 or y2 <= y1:
        print(f"跳过无效框: [{x1},{y1},{x2},{y2}], score={score:.3f}")
        continue

    cv2.rectangle(debug_img_640, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(debug_img_640, f"{score:.2f}", (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    count += 1

print(f"640x640图像上画了 {count} 个框")

# 保存640x640结果
Path("output").mkdir(exist_ok=True)
cv2.imwrite("output/result_640x640.jpg", debug_img_640)

# 把框映射回原图（简单拉伸映射）
# 640x640 -> 1920x1080
scale_x = w_img / 640
scale_y = h_img / 640

debug_img_orig = orig_image.copy()
for i in range(300):
    score = dets[i, 1]
    if score < 0.3:
        continue

    x1, y1, x2, y2 = dets[i, 2:6]

    # 映射回原图
    x1_orig = int(x1 * scale_x)
    y1_orig = int(y1 * scale_y)
    x2_orig = int(x2 * scale_x)
    y2_orig = int(y2 * scale_y)

    cv2.rectangle(debug_img_orig, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 3)
    cv2.putText(debug_img_orig, f"{score:.2f}", (x1_orig, max(y1_orig - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imwrite("output/result_orig_stretch.jpg", debug_img_orig)
print("已保存原图拉伸映射结果")