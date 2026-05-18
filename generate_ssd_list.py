import os

# ===================== 【你的路径 直接粘贴】 =====================
# 训练集
TRAIN_IMAGE_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-train\images"
TRAIN_ANNOT_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\Annotations_train"

# 验证集
VAL_IMAGE_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-val\images"
VAL_ANNOT_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\Annotations_val"

# 输出文件路径（生成在 VisDrone2019 根目录）
OUTPUT_ROOT = r"E:\2026\ultralytics_drone\VisDrone2019"


# ====================================================================

def generate_file_list(img_dir, xml_dir, save_path):
    """
    生成 SSD 所需的 txt 列表
    格式：图片路径 标注路径
    """
    # 获取所有图片文件
    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    lines = []

    for img_name in img_files:
        # 图片完整路径
        img_path = os.path.join(img_dir, img_name)
        # 对应的 XML 文件名
        xml_name = os.path.splitext(img_name)[0] + '.xml'
        xml_path = os.path.join(xml_dir, xml_name)

        # 只保留同时存在图片和标注的条目
        if os.path.exists(xml_path):
            # SSD 格式：用空格分隔 图片路径 和 标注路径
            lines.append(f"{img_path} {xml_path}")

    # 写入文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✅ 生成完成：{save_path}")
    print(f"📊 有效样本数量：{len(lines)}")


if __name__ == '__main__':
    print("=== 开始生成 SSD 训练列表 ===")

    # 生成训练集列表
    train_txt = os.path.join(OUTPUT_ROOT, 'train.txt')
    generate_file_list(TRAIN_IMAGE_DIR, TRAIN_ANNOT_DIR, train_txt)

    print('-' * 50)

    # 生成验证集列表（已修复！）
    val_txt = os.path.join(OUTPUT_ROOT, 'val.txt')
    generate_file_list(VAL_IMAGE_DIR, VAL_ANNOT_DIR, val_txt)

    print("\n🎉 全部生成完成！")
    print(f"训练列表：{train_txt}")
    print(f"验证列表：{val_txt}")