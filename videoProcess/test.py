import os
import glob

# ==================== 修改这里 ====================
# 你的模型所在文件夹（按实际情况改）
model_dir = r"D:\cursor_test\traffic_project\ai_service\weights"

# 或者如果你知道具体文件名，直接写：
# pdmodel_path = r"D:\xxx\model.pdmodel"
# onnx_path = r"D:\xxx\model.onnx"
# =================================================

def check_paddle_input(pdmodel_path):
    import paddle
    from paddle.inference import Config
    params_path = pdmodel_path.replace(".pdmodel", ".pdiparams")
    if not os.path.exists(params_path):
        print(f"❌ 缺少对应的参数文件: {params_path}")
        return
    config = Config(pdmodel_path, params_path)
    predictor = paddle.inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        handle = predictor.get_input_handle(name)
        print(f"✅ Paddle 模型输入: {name} -> 形状: {handle.shape()}")

def check_onnx_input(onnx_path):
    import onnx
    model = onnx.load(onnx_path)
    for input in model.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param for d in input.type.tensor_type.shape.dim]
        print(f"✅ ONNX 模型输入: {input.name} -> 形状: {dims}")

# 自动查找
pdmodel = glob.glob(os.path.join(model_dir, "*.pdmodel"))
onnx_files = glob.glob(os.path.join(model_dir, "*.onnx"))

if pdmodel:
    check_paddle_input(pdmodel[0])
elif onnx_files:
    check_onnx_input(onnx_files[0])
else:
    print("❌ 目录下没找到 .pdmodel 或 .onnx 文件")
    print("💡 提示：RT-DETR 默认输入通常是 [1, 3, 640, 640]")