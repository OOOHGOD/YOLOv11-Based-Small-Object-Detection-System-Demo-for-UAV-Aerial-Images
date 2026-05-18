import os
import cv2
import numpy as np

# ===================== 你的路径 =====================
TRAIN_IMG = r"E:\2026\ultralytics_drone\VisDrone2019-DET-train\images"
TRAIN_LABEL = r"E:\2026\ultralytics_drone\VisDrone2019-DET-train\labels"

# ✅ 已适配：训练用的 10 类（0~9）
class_names = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning_tricycle",
    8: "bus",
    9: "motor"
}

# 检查多少张图片
CHECK_COUNT = 10
# ====================================================


def visualize_one_image(img_name):
    img_path = os.path.join(TRAIN_IMG, img_name)
    txt_path = os.path.join(TRAIN_LABEL, os.path.splitext(img_name)[0] + ".txt")

    if not os.path.exists(txt_path):
        print(f"无标签：{img_name}")
        return

    img = cv2.imread(img_path)
    h_img, w_img = img.shape[:2]

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = list(map(float, line.split()))
        cls_id, cx, cy, nw, nh = parts
        cls_id = int(cls_id)

        # 安全判断：不存在的类别跳过
        if cls_id not in class_names:
            continue

        x1 = int((cx - nw / 2) * w_img)
        y1 = int((cy - nh / 2) * h_img)
        x2 = int((cx + nw / 2) * w_img)
        y2 = int((cy + nh / 2) * h_img)

        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, class_names[cls_id], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow(f"Check: {img_name}", img)
    print(f"✅ 显示：{img_name}，按任意键下一张")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 开始检查
img_list = [f for f in os.listdir(TRAIN_IMG) if f.endswith(('jpg', 'png', 'jpeg'))]
for i, img_name in enumerate(img_list[:CHECK_COUNT]):
    print(f"\n-------- 第 {i+1} 张 --------")
    visualize_one_image(img_name)

cv2.destroyAllWindows()
print("\n🎉 检查完成！")