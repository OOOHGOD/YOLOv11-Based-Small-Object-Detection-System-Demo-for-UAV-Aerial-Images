from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import os
import glob

# ===================== 你的配置 =====================
MODEL_V11 = "yolo11s.pt"
MODEL_PPA = r"E:\2026\ultralytics_drone\ultralytics-main-yolo11-improve\runs\detect\train11s_NeckPPA\weights\best.pt"

TEST_IMAGES_DIR = r"E:\2026\ultralytics_drone\ultralytics-main-yolo11-improve\test_images"
SAVE_DIR = "compare_results"
os.makedirs(SAVE_DIR, exist_ok=True)

SLICE_SIZE = 640
OVERLAP = 0.25
CONF = 0.25
IMG_SIZE = 640
# ====================================================

def plot_yolo(model_path, img_path):
    model = YOLO(model_path)
    res = model(img_path, conf=CONF, imgsz=IMG_SIZE, verbose=False)[0]
    return res.plot()

def plot_sahi(model_path, img_path):
    # SAHI 模型初始化
    detector = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=CONF,
        device="cuda:0",
        image_size=IMG_SIZE
    )
    # 切片推理
    pred = get_sliced_prediction(
        image=img_path,
        detection_model=detector,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP,
        overlap_width_ratio=OVERLAP,
        perform_standard_pred=False,
        verbose=1
    )

    # 读取原图并绘制所有框（修复：这里完整绘制）
    img = cv2.imread(img_path)
    for obj in pred.object_prediction_list:
        # 坐标
        x1, y1 = int(obj.bbox.minx), int(obj.bbox.miny)
        x2, y2 = int(obj.bbox.maxx), int(obj.bbox.maxy)
        # 类别 & 置信度
        cls = obj.category.name
        conf = obj.score.value
        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 标注文字
        cv2.putText(img, f"{cls} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    # 自动加载图片
    img_paths = []
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.JPG"]:
        img_paths += glob.glob(os.path.join(TEST_IMAGES_DIR, ext))

    print(f" 加载图片数量：{len(img_paths)}")

    for idx, img_path in enumerate(img_paths):
        print(f"处理第 {idx+1} 张...")

        # 三行结果
        img1 = plot_yolo(MODEL_V11, img_path)
        img2 = plot_yolo(MODEL_PPA, img_path)
        img3 = plot_sahi(MODEL_PPA, img_path)

        # 拼接
        combined = cv2.hconcat([img1, img2, img3])

        # 标题
        h, w = img1.shape[:2]
        cv2.putText(combined, "YOLOv11", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(combined, "PPA", (w+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(combined, "PPA+SAHI", (2*w+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        save_path = os.path.join(SAVE_DIR, f"compare_{idx+1}.jpg")
        cv2.imwrite(save_path, combined)

    print("\n 全部对比图生成完成！")
    print(f" 结果在：{SAVE_DIR}")