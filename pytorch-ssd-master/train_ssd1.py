# ===================== 超级纯净版：无任何多余依赖 =====================
import os
import logging
import sys

import torch
from torch.utils.data import DataLoader

# 只导入仓库核心代码
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.config import mobilenetv1_ssd_config
from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.data_preprocessing import TrainAugmentation

# ===================== 你的路径（已填好） =====================
YOUR_DATASET_ROOT = r"E:\2026\ultralytics_drone\VisDrone2019"
PRETRAIN_WEIGHTS = r"models/mobilenet_v1_with_relu_69_5.pth"
BATCH_SIZE = 2
EPOCHS = 30
DEVICE = torch.device("cpu")  # 强制CPU，彻底避开CUDA错误


# ===================== 纯净训练函数 =====================
def train(loader, net, criterion, optimizer):
    net.train()
    total_loss = 0
    for images, boxes, labels in loader:
        images, boxes, labels = images.to(DEVICE), boxes.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        conf, loc = net(images)
        reg_loss, cls_loss = criterion(conf, loc, labels, boxes)
        loss = reg_loss + cls_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


# ===================== 主程序 =====================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 数据配置
    cfg = mobilenetv1_ssd_config
    transform = TrainAugmentation(cfg.image_size, cfg.image_mean, cfg.image_std)
    target_transform = MatchPrior(cfg.priors, cfg.center_variance, cfg.size_variance, 0.5)

    # 加载你的VOC数据集
    train_dataset = VOCDataset(YOUR_DATASET_ROOT, transform, target_transform)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)

    # 模型
    net = create_mobilenetv1_ssd(len(train_dataset.class_names))
    net.init_from_base_net(PRETRAIN_WEIGHTS)
    net = net.to(DEVICE)

    # 优化器
    criterion = MultiboxLoss(cfg.priors, device=DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    # 开始训练
    print("✅ 纯净训练开始")
    for epoch in range(EPOCHS):
        loss = train(train_loader, net, criterion, optimizer)
        print(f"Epoch {epoch:2d} | Loss: {loss:.4f}")

        # 保存模型
        if epoch % 5 == 0:
            net.save(f"mb1-ssd-epoch-{epoch}.pth")

    print("🎉 训练完成！")