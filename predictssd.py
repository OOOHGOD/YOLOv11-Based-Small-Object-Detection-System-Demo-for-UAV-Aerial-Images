import os
import torch
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import numpy as np

# --- 1. 配置参数 (必须与训练时一致) ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
WEIGHTS_PATH = r"E:\2026\ultralytics_drone\ssd_results\best_ssd.pth"  # 你的模型路径
IMAGE_PATH = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-val\images\0000023_00300_d_0000009.jpg"  # 待测试图片路径
SAVE_PATH = "prediction_result.jpg"  # 结果保存路径

# 类别列表 (务必与训练时顺序完全一致)
CLASSES = [
    '__background__', 'pedestrian', 'person', 'bicycle', 'car',
    'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'
]
NUM_CLASSES = len(CLASSES)
CONF_THRESHOLD = 0.5  # 置信度阈值，只显示概率大于 0.5 的框


# --- 2. 加载模型 ---
def load_model(weights_path):
    # 初始化 SSD300 VGG16 架构
    model = torchvision.models.detection.ssd300_vgg16(weights=None)
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, NUM_CLASSES)

    # 加载权重
    print(f"Loading weights from {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()  # 切换到推理模式
    return model


# --- 3. 推理与绘图 ---
def predict():
    model = load_model(WEIGHTS_PATH)

    # 读取图片
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    w, h = original_img.size

    # 预处理 (与训练一致：Resize 300x300 并转为 Tensor)
    transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor()
    ])
    img_tensor = transform(original_img).unsqueeze(0).to(DEVICE)  # 增加 Batch 维度

    # 推理
    with torch.no_grad():
        predictions = model(img_tensor)

    # 解析结果 (取第一个 Batch)
    preds = predictions[0]
    boxes = preds['boxes'].cpu().numpy()
    labels = preds['labels'].cpu().numpy()
    scores = preds['scores'].cpu().numpy()

    # 准备绘图
    draw = ImageDraw.Draw(original_img)
    # 尝试加载字体 (如果报错请注释掉)
    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = None

    print(f"Detected objects in {IMAGE_PATH}:")

    count = 0
    for i in range(len(scores)):
        if scores[i] > CONF_THRESHOLD:
            count += 1
            # 坐标转换：由于模型是在 300x300 空间预测的，需要映射回原图大小
            xmin, ymin, xmax, ymax = boxes[i]
            xmin = xmin * w / 300
            ymin = ymin * h / 300
            xmax = xmax * w / 300
            ymax = ymax * h / 300

            # 画框
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

            # 写标签内容
            class_name = CLASSES[labels[i]]
            score_text = f"{class_name} {scores[i]:.2f}"
            draw.text((xmin, ymin - 20), score_text, fill="yellow", font=font)

            print(f" - {class_name}: {scores[i]:.4f}")

    if count == 0:
        print("No objects detected with confidence >", CONF_THRESHOLD)
    else:
        original_img.save(SAVE_PATH)
        print(f"Result saved to {SAVE_PATH}. Total objects: {count}")
        original_img.show()  # 自动弹出图片查看


if __name__ == "__main__":
    # 解决多个 OpenMP 运行库冲突问题
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    predict()