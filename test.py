from ultralytics import YOLO
import torch
import time
import os

# ===================== 【只用改这里！】 =====================
# 把你的 best.pt 路径填在这里
BEST_PT_PATH = r"E:\2026\ultralytics_drone\runs\detect\train11s\weights\best.pt"

# 你的数据集yaml（必须和训练时一样）
DATA_YAML = r"E:\2026\ultralytics_drone\visdrone.yaml"

IMGSZ = 640
BATCH = 8
DEVICE = 0
# ===========================================================

if __name__ == '__main__':
    # ===================== 1. 加载训练好的模型 =====================
    model = YOLO(BEST_PT_PATH)
    print(f"✅ 加载模型成功：{BEST_PT_PATH}")

    # ===================== 2. 测速 FPS =====================
    device = torch.device(f"cuda:{DEVICE}")
    model.model.to(device)  # 把模型放到GPU（修复你之前的报错）

    dummy = torch.randn(1, 3, IMGSZ, IMGSZ).to(device)
    with torch.no_grad():
        # 预热
        for _ in range(10):
            model.model(dummy)
        # 正式测试
        t_start = time.time()
        for _ in range(100):
            model.model(dummy)
        fps = 100 / (time.time() - t_start)

    # ===================== 3. 参数量 =====================
    model_info = model.info()
    params_M = model_info[1] / 1e6

    # ===================== 4. 计算 mAP@0.5 =====================
    print("🔍 正在计算 mAP@0.5...")
    metrics = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        verbose=False
    )
    mAP50 = metrics.box.map50

    # ===================== 5. 保存 TXT =====================
    save_folder = os.path.dirname(BEST_PT_PATH)
    result_file = os.path.join(save_folder, "baseline_result.txt")

    with open(result_file, "w", encoding="utf-8") as f:
        f.write("=== YOLO 模型评估指标 ===\n")
        f.write(f"mAP@0.5: {mAP50:.4f}\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"参数量: {params_M:.2f} M\n")

    # ===================== 6. 打印 =====================
    print("\n" + "="*60)
    print("🎉 评估完成！")
    print(f"📊 mAP@0.5 = {mAP50:.4f}")
    print(f"⚡ FPS     = {fps:.2f}")
    print(f"📦 Params  = {params_M:.2f} M")
    print(f"📂 指标已保存到：{result_file}")
    print("="*60)
