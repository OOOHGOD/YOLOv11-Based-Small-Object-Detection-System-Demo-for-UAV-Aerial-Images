import os
import time
import torch
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torch import nn
from tqdm import tqdm

# 解决库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 配置 ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
WEIGHTS_PATH = "./ssd_result/best_ssd.pth"
NUM_CLASSES = 12
IMAGE_SIZE = 300


def get_model():
    model = torchvision.models.detection.ssd300_vgg16(weights=None)
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, NUM_CLASSES)

    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        print(f"成功加载权重: {WEIGHTS_PATH}")
    else:
        print("警告: 未找到权重文件，将使用随机初始化参数进行测试。")
    return model.to(DEVICE)


# 1. 计算参数量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 30)
    print("模型参数量统计:")
    print(f"总参数量 (Total Params): {total_params / 1e6:.2f} M")
    print(f"可训练参数 (Trainable): {trainable_params / 1e6:.2f} M")
    print("-" * 30)


# 2. 测量 FPS
def measure_fps(model, iterations=100):
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    # 预热
    for _ in range(10):
        _ = model(dummy_input)

    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)

    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    fps = iterations / (time.time() - start_time)
    print(f"推理速度 (FPS): {fps:.2f} frames/sec")
    print(f"单张耗时: {1000 / fps:.2f} ms")
    print("-" * 30)


if __name__ == "__main__":
    model = get_model()
    count_parameters(model)
    measure_fps(model)
