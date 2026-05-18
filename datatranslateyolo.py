import os
from PIL import Image

# ===================== 固定路径（直接用） =====================
BASE_PATH = r"E:\2026\ultralytics_drone"

# 训练集
TRAIN_IMG_DIR = r"E:\2026\ultralytics_drone\VisDrone2019-DET-train\images"
TRAIN_ANNO_DIR = r"E:\2026\ultralytics_drone\VisDrone2019-DET-train\annotations"
TRAIN_LABEL_DIR = r"E:\2026\ultralytics_drone\VisDrone2019-DET-train\labels"

# 验证集
VAL_IMG_DIR = r"E:\2026\ultralytics_drone\VisDrone2019-DET-val\images"
VAL_ANNO_DIR = r"E:\2026\ultralytics_drone\VisDrone2019-DET-val\annotations"
VAL_LABEL_DIR = r"E:\2026\ultralytics_drone\VisDrone2019-DET-val\labels"
# =================================================================


def convert_single(img_path, txt_path, out_path):
    """自动读取图片尺寸，转换单张标签为YOLO格式"""
    try:
        # 自动获取当前图片真实宽高
        with Image.open(img_path) as img:
            w_img, h_img = img.size
    except Exception as e:
        print(f"⚠️  图片读取失败: {img_path}, {e}")
        return

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    yolo_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue

        x = float(parts[0])
        y = float(parts[1])
        w = float(parts[2])
        h = float(parts[3])
        cls = int(parts[5])

        if cls == 0:  # 过滤忽略类
            continue

        # 归一化（自动用当前图片真实尺寸）
        cx = (x + w / 2) / w_img
        cy = (y + h / 2) / h_img
        nw = w / w_img
        nh = h / h_img

        yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(yolo_lines)


def visdrone2yolo_auto(img_dir, anno_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    img_set = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(('jpg', 'png', 'jpeg'))}

    for txt_name in os.listdir(anno_dir):
        if not txt_name.endswith(".txt"):
            continue
        base = os.path.splitext(txt_name)[0]
        if base not in img_set:
            continue

        img_path = os.path.join(img_dir, base + ".jpg")
        txt_path = os.path.join(anno_dir, txt_name)
        out_path = os.path.join(label_dir, txt_name)
        convert_single(img_path, txt_path, out_path)

    print(f"✅ 完成：{label_dir}")


# ===================== 执行转换 =====================
print("🔄 转换训练集...")
visdrone2yolo_auto(TRAIN_IMG_DIR, TRAIN_ANNO_DIR, TRAIN_LABEL_DIR)

print("🔄 转换验证集...")
visdrone2yolo_auto(VAL_IMG_DIR, VAL_ANNO_DIR, VAL_LABEL_DIR)

# ===================== 生成YOLO配置文件 =====================
yaml_path = os.path.join(BASE_PATH, "visdrone.yaml")
yaml_content = f"""path: {BASE_PATH}
train: VisDrone2019-DET-train/images
val: VisDrone2019-DET-val/images
nc: 8
names:
  0: ignore
  1: pedestrian
  2: people
  3: bicycle
  4: car
  5: van
  6: truck
  7: tricycle
"""

with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("\n🎉 全部转换成功！配置文件：", yaml_path)