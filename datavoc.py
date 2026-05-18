import os
import shutil

# ===================== 你的路径 =====================
TRAIN_IMG    = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-train\images"
VAL_IMG      = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-val\images"
TRAIN_XML    = r"E:\2026\ultralytics_drone\VisDrone2019\Annotations_train"
VAL_XML      = r"E:\2026\ultralytics_drone\VisDrone2019\Annotations_val"
ROOT         = r"E:\2026\ultralytics_drone\VisDrone2019"
# ====================================================

# 创建官方VOC目录
JPEG        = os.path.join(ROOT, "JPEGImages")
ANNOT       = os.path.join(ROOT, "Annotations")
IMAGESETS   = os.path.join(ROOT, "ImageSets", "Main")
os.makedirs(JPEG, exist_ok=True)
os.makedirs(ANNOT, exist_ok=True)
os.makedirs(IMAGESETS, exist_ok=True)

# 复制图片
def copy_img(src):
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), os.path.join(JPEG, f))

# 复制XML
def copy_xml(src):
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), os.path.join(ANNOT, f))

# 生成 train.txt / val.txt（只需要文件名，官方要求）
def make_list(img_dir, save_path):
    names = [os.path.splitext(f)[0] for f in os.listdir(img_dir)]
    with open(save_path, "w") as f:
        f.write("\n".join(names))

print("复制图片...")
copy_img(TRAIN_IMG)
copy_img(VAL_IMG)

print("复制XML...")
copy_xml(TRAIN_XML)
copy_xml(VAL_XML)

print("生成 train.txt / val.txt...")
make_list(TRAIN_IMG, os.path.join(IMAGESETS, "train.txt"))
make_list(VAL_IMG, os.path.join(IMAGESETS, "val.txt"))

print("\n✅ 全部完成！现在你的数据集是官方 SSD 可直接训练的 VOC 格式！")