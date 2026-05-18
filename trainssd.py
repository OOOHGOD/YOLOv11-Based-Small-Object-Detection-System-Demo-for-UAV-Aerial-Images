import os
import xml.etree.ElementTree as ET
import torch
import torch.utils.data as data
from PIL import Image
import torchvision
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import SSD300_VGG16_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ================= 1. 配置参数 =================
DATA_ROOT = r"E:\2026\ultralytics_drone\datasets\VOC2007"  # 你的数据集根目录
# VisDrone 标准 10 类 + 背景
CLASSES = [
    '__background__', 'pedestrian', 'person', 'bicycle', 'car',
    'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others'
]
NUM_CLASSES = len(CLASSES)
BATCH_SIZE = 16  # 根据显存调整
EPOCHS = 100
LR = 1e-4  # 较低的学习率防止 NaN
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SAVE_DIR = "ssd_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ================= 2. 增强型数据加载类 =================
class VisDroneDataset(data.Dataset):
    def __init__(self, root, image_set="train"):
        self.root = root
        self.img_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations")

        # 获取候选 ID
        txt_path = os.path.join(root, f"ImageSets/Main/{image_set}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                raw_ids = [line.strip() for line in f.readlines()]
        else:
            raw_ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir)]

        print(f"Checking and filtering {image_set} annotations...")
        self.ids = []
        for img_id in tqdm(raw_ids):
            ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
            if not os.path.exists(ann_path): continue

            # 预扫描：确保图片中至少有一个有效框
            tree = ET.parse(ann_path)
            root_xml = tree.getroot()
            valid = False
            for obj in root_xml.findall("object"):
                if obj.find("name").text in CLASSES:
                    valid = True
                    break
            if valid: self.ids.append(img_id)

        print(f"Keep {len(self.ids)} images with valid targets (Total: {len(raw_ids)}).")

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        tree = ET.parse(os.path.join(self.ann_dir, f"{img_id}.xml"))
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASSES: continue

            bndbox = obj.find("bndbox")
            # 关键：读取并裁剪坐标，防止超出图片边界
            xmin = max(0, float(bndbox.find("xmin").text))
            ymin = max(0, float(bndbox.find("ymin").text))
            xmax = min(width, float(bndbox.find("xmax").text))
            ymax = min(height, float(bndbox.find("ymax").text))

            # 过滤无效框
            if xmax > xmin + 1 and ymax > ymin + 1:
                # SSD 需要将坐标缩放到 [0, 300] 对应图片尺寸
                boxes.append([
                    xmin * 300.0 / width,
                    ymin * 300.0 / height,
                    xmax * 300.0 / width,
                    ymax * 300.0 / height
                ])
                labels.append(CLASSES.index(name))

        # 转换为 Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 处理异常：如果该图确实没目标（虽理论上已过滤）
        if boxes.shape[0] == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        # 预处理
        img = torchvision.transforms.functional.to_tensor(img)
        img = torchvision.transforms.functional.resize(img, [300, 300])

        return img, target

    def __len__(self):
        return len(self.ids)


def collate_fn(batch):
    return tuple(zip(*batch))


# ================= 3. 构建模型 =================
def get_model(num_classes):
    model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    # 替换分类头
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
    return model


# ================= 4. 训练主程序 =================
def main():
    print(f"Training on: {DEVICE}")

    train_ds = VisDroneDataset(DATA_ROOT, "train")
    val_ds = VisDroneDataset(DATA_ROOT, "val")
    train_loader = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = get_model(NUM_CLASSES).to(DEVICE)
    # 使用 Adam 处理不稳定的梯度
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": []}
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, targets in pbar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                print(f"\nLoss is {losses.item()}, skipping batch...")
                continue

            optimizer.zero_grad()
            losses.backward()

            # --- 关键：梯度裁剪，防止 NaN ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            train_loss += losses.item()
            pbar.set_postfix(loss=losses.item())

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # 验证阶段
        model.train()  # 保持 train 模式以获取 loss
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                v_loss_dict = model(images, targets)
                v_losses = sum(v for v in v_loss_dict.values())
                val_loss += v_losses.item()

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        print(f"Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 保存模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_ssd.pth"))
            print(">>> Best Model Saved!")

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_ssd.pth"))

    # ================= 5. 结果可视化 =================
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label='Train Loss')
    plt.plot(history["val_loss"], label='Val Loss')
    plt.title('SSD Training Loss (VisDrone2019)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "training_result.png"))
    print(f"Training Complete. Curves saved in {SAVE_DIR}")


if __name__ == "__main__":
    main()