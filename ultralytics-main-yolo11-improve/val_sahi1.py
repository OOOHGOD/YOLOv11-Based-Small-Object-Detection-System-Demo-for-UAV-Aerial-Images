import os
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from ultralytics.utils.metrics import ap_per_class, ConfusionMatrix

# =========================
# 1. 路径配置
# =========================
MODEL_PATH = r"runs\detect\train11s_NeckPPA\weights\best.pt"
DATA_YAML  = r"ultralytics\cfg\datasets\VisDrone.yaml"

# =========================
# 2. 实验网格配置（阶段1参数）
# =========================
#实验参数1
"""
SLICE_SIZE = 640
OVERLAP_RATIO = 0.25
INFER_CONF_THRESHOLD = 0.001
EXPORT_CONF_THRESHOLD = 0.1
IMG_SIZE = 640
DEVICE = "cuda:0"
"""
#实验参数2
SLICE_SIZE = 640           # 保持不变
OVERLAP_RATIO = 0.4        # 小目标必须提高重叠
INFER_CONF_THRESHOLD = 0.02  # 过滤冗余框
EXPORT_CONF_THRESHOLD = 0.15  # 提高置信度，减少FP
POSTPROCESS_MATCH_THRESHOLDS = (0.25, 0.3, 0.35)  # 真正的最优区间
POSTPROCESS_TYPE = "GREEDYNMM"  # 最佳后处理
# 网格扫参范围
POSTPROCESS_TYPES = ("NMS", "NMM", "GREEDYNMM")
POSTPROCESS_MATCH_METRIC = "IOU"
POSTPROCESS_MATCH_THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
POSTPROCESS_CLASS_AGNOSTIC = False
# ====================================================

def image_to_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    lower_parts = [p.lower() for p in parts]
    if "images" in lower_parts:
        i = lower_parts.index("images")
        parts[i] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")

def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    lines = [x.strip() for x in label_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    cls = []
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        x1 = (x - w / 2.0) * img_w
        y1 = (y - h / 2.0) * img_h
        x2 = (x + w / 2.0) * img_w
        y2 = (y + h / 2.0) * img_h
        boxes.append([x1, y1, x2, y2])
        cls.append(c)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(cls, dtype=np.int64)

def compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    from ultralytics.utils.metrics import box_iou
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    return box_iou(boxes1, boxes2)

def match_predictions(
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    conf: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    iouv: np.ndarray,
) -> np.ndarray:
    n_pred = pred_boxes.shape[0]
    n_iou = iouv.shape[0]
    tp = np.zeros((n_pred, n_iou), dtype=bool)
    if n_pred == 0 or gt_boxes.shape[0] == 0:
        return tp

    order = np.argsort(-conf)
    pred_boxes = pred_boxes[order]
    pred_cls = pred_cls[order]
    conf = conf[order]

    iou = compute_iou_matrix(torch.from_numpy(pred_boxes), torch.from_numpy(gt_boxes)).numpy()

    for t, thr in enumerate(iouv):
        matched_gt = set()
        for i in range(n_pred):
            candidates = np.where(gt_cls == pred_cls[i])[0]
            if candidates.size == 0:
                continue
            ious = iou[i, candidates]
            j = int(candidates[int(np.argmax(ious))])
            if iou[i, j] >= thr and j not in matched_gt:
                tp[i, t] = True
                matched_gt.add(j)

    out = np.zeros((n_pred, n_iou), dtype=bool)
    out[order] = tp
    return out

def run_one_setting(
    img_paths: list[Path],
    names: dict,
    nc: int,
    repo_root: Path,
    model_path: Path,
    save_dir: Path,
    postprocess_type: str,
    match_threshold: float,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=INFER_CONF_THRESHOLD,
        device=DEVICE,
        image_size=IMG_SIZE,
    )

    predictions_eval = []
    predictions_export = []
    tp_all = []
    conf_all = []
    pred_cls_all = []
    target_cls_all = []
    total_targets = 0
    matched_eval_50 = 0
    matched_export_50 = 0
    iouv = np.linspace(0.5, 0.95, 10, dtype=np.float32)

    # Initialize Ultralytics ConfusionMatrix
    confusion_matrix = ConfusionMatrix(nc=nc, conf=EXPORT_CONF_THRESHOLD, iou_thres=0.45)

    for img_path in tqdm(img_paths, desc=f"{postprocess_type}-thr{match_threshold}", leave=False):
        result = get_sliced_prediction(
            image=str(img_path),
            detection_model=detection_model,
            slice_height=SLICE_SIZE,
            slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            perform_standard_pred=False,
            postprocess_type=postprocess_type,
            postprocess_match_metric=POSTPROCESS_MATCH_METRIC,
            postprocess_match_threshold=float(match_threshold),
            postprocess_class_agnostic=bool(POSTPROCESS_CLASS_AGNOSTIC),
        )

        img = Image.open(str(img_path))
        width, height = img.size
        img_id = img_path.stem

        gt_boxes, gt_cls = load_yolo_labels(image_to_label_path(img_path), width, height)
        if gt_cls.size:
            target_cls_all.append(gt_cls)
            total_targets += int(gt_cls.size)

        pred_boxes = []
        pred_cls = []
        pred_conf = []
        items = []

        for obj in result.object_prediction_list:
            x1 = obj.bbox.minx
            y1 = obj.bbox.miny
            x2 = obj.bbox.maxx
            y2 = obj.bbox.maxy
            conf = float(obj.score.value)
            cls = int(obj.category.id)

            pred_boxes.append([x1, y1, x2, y2])
            pred_cls.append(cls)
            pred_conf.append(conf)

            items.append(
                {
                    "image_id": img_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": conf,
                    "category_id": cls,
                }
            )

        pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
        pred_cls = np.asarray(pred_cls, dtype=np.int64)
        pred_conf = np.asarray(pred_conf, dtype=np.float32)

        # 修正 1-based class ID (如果是从1开始的则减1)
        if nc > 0 and pred_cls.size and pred_cls.min() >= 1 and pred_cls.max() <= nc and pred_cls.max() == nc:
            pred_cls = pred_cls - 1
            for it in items:
                it["category_id"] = int(it["category_id"]) - 1

        if nc > 0 and pred_cls.size:
            keep = (pred_cls >= 0) & (pred_cls < nc)
            pred_boxes = pred_boxes[keep]
            pred_cls = pred_cls[keep]
            pred_conf = pred_conf[keep]

        for it in items:
            predictions_eval.append(it)
            if float(it["score"]) >= EXPORT_CONF_THRESHOLD:
                predictions_export.append(it)

        # 匹配 TP/FP
        tp = match_predictions(pred_boxes, pred_cls, pred_conf, gt_boxes, gt_cls, iouv)
        if pred_conf.size:
            tp_all.append(tp)
            conf_all.append(pred_conf)
            pred_cls_all.append(pred_cls)
            matched_eval_50 += int(tp[:, 0].sum()) if tp.size else 0

        if pred_conf.size:
            export_mask = pred_conf >= float(EXPORT_CONF_THRESHOLD)
            if np.any(export_mask):
                tp_export = match_predictions(
                    pred_boxes[export_mask],
                    pred_cls[export_mask],
                    pred_conf[export_mask],
                    gt_boxes,
                    gt_cls,
                    iouv,
                )
                matched_export_50 += int(tp_export[:, 0].sum()) if tp_export.size else 0

        # Update Confusion Matrix (使用导出阈值的数据来更新混淆矩阵比较合理)
        if gt_boxes.size > 0:
            gt_t = torch.from_numpy(gt_boxes)
            gt_c = torch.from_numpy(gt_cls)
        else:
            gt_t = torch.zeros((0, 4))
            gt_c = torch.zeros((0,))
        
        if pred_boxes.size > 0:
            pd_t = torch.from_numpy(np.concatenate([pred_boxes, pred_conf[:, None], pred_cls[:, None]], axis=1))
        else:
            pd_t = torch.zeros((0, 6))
            
        confusion_matrix.process_batch(pd_t, gt_t, gt_c)

    # 导出 JSON
    pred_eval_json = save_dir / "predictions_eval.json"
    with open(pred_eval_json, "w", encoding="utf-8") as f:
        json.dump(predictions_eval, f)

    pred_json = save_dir / "predictions.json"
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(predictions_export, f)

    # 绘图和计算 AP
    tp = np.concatenate(tp_all, axis=0) if tp_all else np.zeros((0, iouv.size), dtype=bool)
    conf = np.concatenate(conf_all, axis=0) if conf_all else np.zeros((0,), dtype=np.float32)
    pred_cls = np.concatenate(pred_cls_all, axis=0) if pred_cls_all else np.zeros((0,), dtype=np.int64)
    target_cls = np.concatenate(target_cls_all, axis=0) if target_cls_all else np.zeros((0,), dtype=np.int64)

    # 核心：调用 Ultralytics ap_per_class 并 plot=True，会自动保存 PR/P/R/F1 曲线
    out = ap_per_class(tp, conf, pred_cls, target_cls, plot=True, save_dir=save_dir, names=names)
    ap = out[5] if isinstance(out, tuple) and len(out) > 5 else np.zeros((0, 10), dtype=np.float32)

    map50 = float(ap[:, 0].mean()) if hasattr(ap, "size") and ap.size else 0.0
    map5095 = float(ap.mean()) if hasattr(ap, "size") and ap.size else 0.0
    recall50_eval = float(matched_eval_50 / total_targets) if total_targets else 0.0
    recall50_export = float(matched_export_50 / total_targets) if total_targets else 0.0

    # 绘制/保存混淆矩阵
    confusion_matrix.plot(save_dir=save_dir, names=tuple(names.values()) if names else ())

    metrics = {
        "mAP50": map50,
        "mAP50-95": map5095,
        "num_images": int(len(img_paths)),
        "num_predictions": int(conf.shape[0]),
        "num_targets": int(target_cls.shape[0]),
        "recall50_eval": recall50_eval,
        "recall50_export": recall50_export,
        "infer_conf_threshold": float(INFER_CONF_THRESHOLD),
        "export_conf_threshold": float(EXPORT_CONF_THRESHOLD),
        "slice_size": int(SLICE_SIZE),
        "overlap_ratio": float(OVERLAP_RATIO),
        "imgsz": int(IMG_SIZE),
        "model_path": str(model_path),
        "data_yaml": str(Path(DATA_YAML).resolve() if Path(DATA_YAML).is_absolute() else (repo_root / DATA_YAML).resolve()),
        "postprocess_type": str(postprocess_type),
        "postprocess_match_metric": str(POSTPROCESS_MATCH_METRIC),
        "postprocess_match_threshold": float(match_threshold),
        "postprocess_class_agnostic": bool(POSTPROCESS_CLASS_AGNOSTIC),
    }

    with open(save_dir / "metrics_sahi.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent
    runs_dir = (repo_root / "runs").resolve()
    base_dir = (runs_dir / "val" / "val_sahi_grid").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(DATA_YAML)
    root = Path(data["path"])
    val_src = data.get("val")
    val_roots = [Path(x) for x in val_src] if isinstance(val_src, (list, tuple)) else [Path(val_src)]
    img_paths: list[Path] = []
    for r in val_roots:
        r = r if r.is_absolute() else root / r
        if r.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"):
                img_paths.extend(list(r.rglob(ext)))
        elif r.is_file() and r.suffix.lower() == ".txt":
            img_paths.extend([Path(x.strip()) for x in r.read_text(encoding="utf-8").splitlines() if x.strip()])
        elif r.is_file():
            img_paths.append(r)
    img_paths = sorted(img_paths)

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    if not isinstance(names, dict):
        names = {}
    nc = len(names) if names else int(data.get("nc", 0) or 0)

    model_path = Path(MODEL_PATH)
    if not model_path.is_absolute():
        model_path = (repo_root / model_path).resolve()

    print(f"✅ 验证集图片数量: {len(img_paths)}")
    print(f"✅ base_dir: {base_dir}")
    print(f"slice={SLICE_SIZE} overlap={OVERLAP_RATIO} infer_conf={INFER_CONF_THRESHOLD} export_conf={EXPORT_CONF_THRESHOLD}")
    print(f"postprocess_types={POSTPROCESS_TYPES} match_metric={POSTPROCESS_MATCH_METRIC} thresholds={POSTPROCESS_MATCH_THRESHOLDS} class_agnostic={POSTPROCESS_CLASS_AGNOSTIC}")

    summary = []
    for ptype in POSTPROCESS_TYPES:
        for thr in POSTPROCESS_MATCH_THRESHOLDS:
            tag = f"s{SLICE_SIZE}_ov{OVERLAP_RATIO}_{ptype}_m{POSTPROCESS_MATCH_METRIC}_t{thr}_ca{int(POSTPROCESS_CLASS_AGNOSTIC)}"
            save_dir = base_dir / tag
            metrics = run_one_setting(img_paths, names, nc, repo_root, model_path, save_dir, ptype, float(thr))
            summary.append(metrics)
            print(
                f"[{tag}] mAP50={metrics['mAP50']:.4f} R@0.5(eval)={metrics['recall50_eval']:.4f} "
                f"pred={metrics['num_predictions']}"
            )

    with open(base_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n🚀 12组网格实验全部完成！")
    print(f"所有可视化图表（P_curve, R_curve, PR_curve, F1_curve, confusion_matrix）和结果JSON均保存在: {base_dir}")
