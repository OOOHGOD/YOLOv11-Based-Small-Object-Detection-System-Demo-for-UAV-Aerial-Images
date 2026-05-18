import os
import cv2
import json
from tqdm import tqdm


def visdrone_to_coco(image_dir, label_dir, save_path):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "ignored_regions"},  # VisDrone 默认 0 是忽略区
            {"id": 1, "name": "pedestrian"},
            {"id": 2, "name": "people"},
            {"id": 3, "name": "bicycle"},
            {"id": 4, "name": "car"},
            {"id": 5, "name": "van"},
            {"id": 6, "name": "truck"},
            {"id": 7, "name": "tricycle"},
            {"id": 8, "name": "awning-tricycle"},
            {"id": 9, "name": "bus"},
            {"id": 10, "name": "motor"}
        ]
    }

    ann_id = 0
    image_id = 0

    img_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in tqdm(img_list, desc="Converting VisDrone to COCO"):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # 添加图片信息
        coco_format["images"].append({
            "file_name": img_name,
            "id": image_id,
            "width": width,
            "height": height
        })

        # 读取对应的 txt 标签
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_name)

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = list(map(int, line.strip().split(',')))
                    if len(parts) < 8: continue

                    # VisDrone 格式: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                    x, y, w, h, score, category, _, _ = parts

                    # 过滤掉不需要的类别 (如 0 和 11)
                    if category == 0 or category == 11:
                        continue

                    coco_format["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    })
                    ann_id += 1

        image_id += 1

    with open(save_path, "w") as f:
        json.dump(coco_format, f)
    print(f"转换完成！JSON 已保存至: {save_path}")


# --- 执行转换 ---
VAL_IMG = r"E:\2026\ultralytics_drone\VisDrone2019-DET-val\images"
VAL_LAB = r"E:\2026\ultralytics_drone\VisDrone2019-DET-val\annotations"  # 这里放原厂的txt
SAVE_JSON = r"E:\2026\ultralytics_drone\instances_val.json"

visdrone_to_coco(VAL_IMG, VAL_LAB, SAVE_JSON)