import os
import cv2
import xml.etree.ElementTree as ET

# ===================== 【你的路径 直接用】 =====================
# 训练集路径（核对训练集）
IMG_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-train\images"
XML_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\Annotations_train"

# 验证集路径（想核对验证集就把上面两行注释，打开下面两行）
# IMG_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\VisDrone2019-DET-val\images"
# XML_DIR = r"E:\2026\ultralytics_drone\VisDrone2019\Annotations_val"

# 类别（和转换时一致）
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


# ==============================================================

def visualize_one_image(img_path, xml_path):
    """读取单张图片和XML，画出框"""
    img = cv2.imread(img_path)
    if img is None:
        print("找不到图片：", img_path)
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 遍历所有目标
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        bndbox = obj.find("bndbox")

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        # 画框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, cls_name, (xmin, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 显示
    cv2.imshow("VisDrone XML 标注核对", img)
    print("✅ 显示图片：", os.path.basename(img_path))
    print("按 空格 下一张 | 按 ESC 退出")

    key = cv2.waitKey(0)
    if key == 27:  # ESC退出
        cv2.destroyAllWindows()
        exit()


def batch_check():
    """批量浏览所有图片"""
    xml_files = [f for f in os.listdir(XML_DIR) if f.endswith(".xml")]
    print(f"找到 {len(xml_files)} 个标注文件，开始核对...")

    for xml_file in xml_files:
        img_name = os.path.splitext(xml_file)[0] + ".jpg"
        img_path = os.path.join(IMG_DIR, img_name)
        xml_path = os.path.join(XML_DIR, xml_file)

        if not os.path.exists(img_path):
            continue

        visualize_one_image(img_path, xml_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== VisDrone XML 标签可视化核对工具 ===")
    batch_check()