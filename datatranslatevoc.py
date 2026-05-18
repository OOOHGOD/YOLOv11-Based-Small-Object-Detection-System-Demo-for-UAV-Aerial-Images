import os
import cv2
from xml.dom import minidom

# ===================== 【你的数据集路径 直接粘贴】 =====================
# 训练集
TRAIN_IMG_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-train\images"
TRAIN_ANNOT_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-train\annotations"

# 验证集
VAL_IMG_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-val\images"
VAL_ANNOT_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-val\annotations"

# 输出 XML 文件夹（自动创建）
TRAIN_XML_OUT = r"E:\2026\ultralytics_drone\VisDrone2019\Annotations_train"
VAL_XML_OUT = r"E:\2026\ultralytics_drone\VisDrone2019\Annotations_val"

# VisDrone 官方类别（必须和 SSD 配置对应）
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]
# ====================================================================

def create_xml_annotation(img_path, xml_save_path, bboxes, classes):
    """生成 SSD 可用的 VOC 格式 XML"""
    img = cv2.imread(img_path)
    h, w, c = img.shape

    doc = minidom.Document()
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    # 图片信息
    folder = doc.createElement('folder')
    folder.appendChild(doc.createTextNode('VisDrone'))
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(os.path.basename(img_path)))
    annotation.appendChild(filename)

    # 尺寸
    size = doc.createElement('size')
    annotation.appendChild(size)

    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(w)))
    size.appendChild(width)

    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(h)))
    size.appendChild(height)

    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(c)))
    size.appendChild(depth)

    # 目标框
    for cls, (x1, y1, x2, y2) in zip(classes, bboxes):
        obj = doc.createElement('object')
        annotation.appendChild(obj)

        name = doc.createElement('name')
        name.appendChild(doc.createTextNode(cls))
        obj.appendChild(name)

        bndbox = doc.createElement('bndbox')
        obj.appendChild(bndbox)

        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(x1)))
        bndbox.appendChild(xmin)

        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(y1)))
        bndbox.appendChild(ymin)

        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(x2)))
        bndbox.appendChild(xmax)

        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(y2)))
        bndbox.appendChild(ymax)

    # 保存 XML
    with open(xml_save_path, 'w', encoding='utf-8') as f:
        doc.writexml(f, indent='\t', addindent='\t', newl='\n', encoding='utf-8')

def convert_one_folder(img_dir, annot_dir, xml_out_dir):
    os.makedirs(xml_out_dir, exist_ok=True)
    annot_files = [f for f in os.listdir(annot_dir) if f.endswith('.txt')]

    for txt_file in annot_files:
        img_name = os.path.splitext(txt_file)[0] + '.jpg'
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(annot_dir, txt_file)
        xml_path = os.path.join(xml_out_dir, os.path.splitext(txt_file)[0] + '.xml')

        if not os.path.exists(img_path):
            continue

        bboxes = []
        classes = []

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            data = line.split(',')
            if len(data) < 8:
                continue

            # VisDrone 格式：x,y,w,h,score,category,truncation,occlusion
            x = int(float(data[0]))
            y = int(float(data[1]))
            w = int(float(data[2]))
            h = int(float(data[3]))
            cat_id = int(data[5]) - 1  # 从 0 开始

            if 0 <= cat_id < len(CLASS_NAMES):
                cls_name = CLASS_NAMES[cat_id]
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                bboxes.append((x1, y1, x2, y2))
                classes.append(cls_name)

        if bboxes:
            create_xml_annotation(img_path, xml_path, bboxes, classes)

    print(f"✅ 转换完成：{xml_out_dir}")

if __name__ == '__main__':
    print("开始转换训练集...")
    convert_one_folder(TRAIN_IMG_DIR, TRAIN_ANNOT_DIR, TRAIN_XML_OUT)

    print("\n开始转换验证集...")
    convert_one_folder(VAL_IMG_DIR, VAL_ANNOT_DIR, VAL_XML_OUT)

    print("\n🎉 全部完成！SSD 可以直接使用 XML 标注训练")