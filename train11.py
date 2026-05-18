from ultralytics import YOLO
import torch
import time
import os

# ===================== 固定配置 =====================
DATA_YAML = r"E:\2026\ultralytics_drone\visdrone.yaml"
MODEL = "yolo11s.pt"
EPOCHS = 100
IMGSZ = 640
BATCH = 8
DEVICE = 0
NAME = "train11s"
OPTIMIZER = "SGD"
# =====================================================


if __name__ == '__main__':

    # 1. 加载原始模型
    model = YOLO(MODEL)
    print(" 已加载 YOLO11s 官方预训练权重")

    # 2. 开始训练（完全按你的要求）
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        name=NAME,
        optimizer=OPTIMIZER,
        patience=0,
        save=True,
        val=True,
        plots=True,
        amp=True,
    )

    # 3. 加载训练好的最优模型
    best_model = YOLO(model.trainer.best)
    print(f"\n 最优模型路径：{model.trainer.best}")

    # 4. 测速 FPS
    dummy = torch.randn(1, 3, IMGSZ, IMGSZ).to(f"cuda:{DEVICE}")
    with torch.no_grad():
        for _ in range(10):
            best_model.model(dummy)
        t0 = time.time()
        for _ in range(100):
            best_model.model(dummy)
        fps = 100 / (time.time() - t0)

    # 5. 参数量
    model_info = best_model.info()
    params_M = model_info[1] / 1e6

    # 6. 计算 mAP@0.5
    metrics = best_model.val(data=DATA_YAML, imgsz=IMGSZ, batch=BATCH, device=DEVICE)
    mAP50 = metrics.box.map50

    # 7. 保存 Baseline 结果
    save_folder = model.trainer.save_dir
    result_file = os.path.join(save_folder, "baseline_result.txt")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=== YOLO11s Baseline 训练结果 ===\n")
        f.write(f"模型: YOLO11s\n")
        f.write(f"数据集: VisDrone2019\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch: {BATCH}\n")
        f.write(f"Optimizer: {OPTIMIZER}\n\n")
        f.write(f"mAP@0.5: {mAP50:.4f}\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"参数量: {params_M:.2f} M\n")

    # 打印结果
    print("\n" + "="*60)
    print("🎉 YOLO11s 基础训练完成！")
    print(f"📊 mAP@0.5 = {mAP50:.4f}")
    print(f"⚡ FPS     = {fps:.2f}")
    print(f"📦 Params  = {params_M:.2f} M")
    print(f"📂 结果保存到: {save_folder}")
    print("="*60)
