import ultralytics
import torch

# 打印基础信息
print("Hello World!")

# 正确获取版本号的方式：通过模块名访问
print(f"Ultralytics version: {ultralytics.__version__}")
print(f"PyTorch version: {torch.__version__}")

# 检查 GPU 环境
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU instead.")
