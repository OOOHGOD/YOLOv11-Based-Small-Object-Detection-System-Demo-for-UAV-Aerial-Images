import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead


CLASSES = [
    "__background__",
    "pedestrian",
    "person",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
]


class VOCLikeDataset300(data.Dataset):
    def __init__(self, root: str, image_set: str = "val", keep_empty: bool = False):
        self.root = str(root)
        self.img_dir = os.path.join(self.root, "JPEGImages")
        self.ann_dir = os.path.join(self.root, "Annotations")

        txt_path = os.path.join(self.root, f"ImageSets/Main/{image_set}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_ids = [line.strip() for line in f.readlines() if line.strip()]
        else:
            raw_ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir)]

        self.ids = []
        for img_id in raw_ids:
            ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
            if not os.path.exists(ann_path):
                continue
            if keep_empty:
                self.ids.append(img_id)
                continue

            tree = ET.parse(ann_path)
            root_xml = tree.getroot()
            valid = False
            for obj in root_xml.findall("object"):
                name = obj.find("name").text
                if name in CLASSES and name != "__background__":
                    valid = True
                    break
            if valid:
                self.ids.append(img_id)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASSES or name == "__background__":
                continue
            bndbox = obj.find("bndbox")
            xmin = max(0.0, float(bndbox.find("xmin").text))
            ymin = max(0.0, float(bndbox.find("ymin").text))
            xmax = min(float(width), float(bndbox.find("xmax").text))
            ymax = min(float(height), float(bndbox.find("ymax").text))

            if xmax > xmin + 1 and ymax > ymin + 1:
                boxes.append(
                    [
                        xmin * 300.0 / float(width),
                        ymin * 300.0 / float(height),
                        xmax * 300.0 / float(width),
                        ymax * 300.0 / float(height),
                    ]
                )
                labels.append(CLASSES.index(name))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": img_id}

        img = torchvision.transforms.functional.to_tensor(img)
        img = torchvision.transforms.functional.resize(img, [300, 300])
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes: int, pretrained_backbone: bool = True):
    weights = SSD300_VGG16_Weights.DEFAULT if pretrained_backbone else None
    model = torchvision.models.detection.ssd300_vgg16(weights=weights)
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
    return model


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    x11, y11, x12, y12 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    xi1 = np.maximum(x11, x21)
    yi1 = np.maximum(y11, y21)
    xi2 = np.minimum(x12, x22)
    yi2 = np.minimum(y12, y22)
    inter_w = np.maximum(0.0, xi2 - xi1)
    inter_h = np.maximum(0.0, yi2 - yi1)
    inter = inter_w * inter_h
    area1 = np.maximum(0.0, x12 - x11) * np.maximum(0.0, y12 - y11)
    area2 = np.maximum(0.0, x22 - x21) * np.maximum(0.0, y22 - y21)
    union = area1 + area2 - inter + 1e-9
    return (inter / union).astype(np.float32)


def ap_from_pr(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return float("nan")
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    y = np.interp(x, mrec, mpre)
    return float(np.trapz(y, x))


def precision_recall_curve_from_detections(
    predictions: list,
    gt_by_image: dict,
    iou_thres: float,
):
    if len(predictions) == 0:
        return np.array([]), np.array([]), np.array([])

    predictions = sorted(predictions, key=lambda x: float(x[1]), reverse=True)
    gt_used = {k: np.zeros(len(v), dtype=bool) for k, v in gt_by_image.items()}
    scores = np.zeros(len(predictions), dtype=np.float32)
    tp = np.zeros(len(predictions), dtype=np.float32)
    fp = np.zeros(len(predictions), dtype=np.float32)

    for i, (img_key, score, pbox) in enumerate(predictions):
        scores[i] = float(score)
        gts = gt_by_image.get(img_key, np.zeros((0, 4), dtype=np.float32))
        used = gt_used.get(img_key, np.zeros((0,), dtype=bool))
        if gts.shape[0] == 0:
            fp[i] = 1.0
            continue
        ious = box_iou_np(np.asarray([pbox], dtype=np.float32), gts)[0]
        best = int(np.argmax(ious))
        if float(ious[best]) >= float(iou_thres) and not bool(used[best]):
            tp[i] = 1.0
            used[best] = True
            gt_used[img_key] = used
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    npos = int(sum(v.shape[0] for v in gt_by_image.values()))
    if npos == 0:
        return scores, np.array([]), np.array([])
    recall = tp_cum / float(npos)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return scores, recall, precision


def make_output_dir(out_dir: str) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "pr_curves").mkdir(parents=True, exist_ok=True)
    (out_path / "viz2").mkdir(parents=True, exist_ok=True)
    return out_path


def plot_curve(x, y, title, xlabel, ylabel, save_path: Path):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_pr_curve_per_class(class_name: str, recall: np.ndarray, precision: np.ndarray, ap: float, save_path: Path):
    plt.figure(figsize=(7, 6))
    if recall.size:
        plt.plot(recall, precision, linewidth=2)
    plt.title(f"PR Curve: {class_name} (AP@0.5={ap:.3f})" if not math.isnan(ap) else f"PR Curve: {class_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_ap_bar(class_names: list[str], ap_list: list[float], save_path: Path):
    ap_vals = np.array(ap_list, dtype=np.float32)
    valid = np.isfinite(ap_vals)
    order = np.argsort(np.where(valid, ap_vals, -1.0))[::-1]
    names_sorted = [class_names[i] for i in order]
    ap_sorted = ap_vals[order]

    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(names_sorted)), ap_sorted)
    plt.xticks(np.arange(len(names_sorted)), names_sorted, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("AP@0.5")
    plt.title("AP by class")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, names: list[str], save_path: Path, normalize: bool):
    cm_plot = cm.astype(np.float32)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = cm_plot / np.maximum(row_sums, 1e-9)

    plt.figure(figsize=(10, 9))
    plt.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45, ha="right")
    plt.yticks(tick_marks, names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def greedy_match_by_iou(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thres: float):
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return [], set(range(pred_boxes.shape[0])), set(range(gt_boxes.shape[0]))
    iou = box_iou_np(pred_boxes, gt_boxes)
    pi, gi = np.where(iou >= float(iou_thres))
    if pi.size == 0:
        return [], set(range(pred_boxes.shape[0])), set(range(gt_boxes.shape[0]))
    pairs = list(zip(pi.tolist(), gi.tolist(), iou[pi, gi].tolist()))
    pairs.sort(key=lambda x: float(x[2]), reverse=True)
    matched_p = set()
    matched_g = set()
    matches = []
    for p, g, v in pairs:
        if p in matched_p or g in matched_g:
            continue
        matched_p.add(p)
        matched_g.add(g)
        matches.append((p, g, float(v)))
    unmatched_p = set(range(pred_boxes.shape[0])) - matched_p
    unmatched_g = set(range(gt_boxes.shape[0])) - matched_g
    return matches, unmatched_p, unmatched_g


@torch.inference_mode()
def run_eval(
    weights_path: str,
    data_root: str,
    split: str,
    out_dir: str,
    batch_size: int,
    num_workers: int,
    iou_thres: float,
    cm_conf_thres: float,
    pred_min_conf: float,
    device: str,
):
    out_path = make_output_dir(out_dir)

    ds = VOCLikeDataset300(data_root, split)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    model = get_model(len(CLASSES), pretrained_backbone=False)
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()

    class_names = CLASSES[1:]
    n_cls = len(class_names)
    bg_i = n_cls

    gt_boxes_by_class = {c: {} for c in range(1, len(CLASSES))}
    preds_by_class = {c: [] for c in range(1, len(CLASSES))}
    all_gt = {}
    all_preds = {}

    for images, targets in tqdm(dl, desc="SSD eval", ncols=100):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            img_id = tgt["image_id"]

            gt_boxes = tgt["boxes"].cpu().numpy().astype(np.float32)
            gt_labels = tgt["labels"].cpu().numpy().astype(np.int64)

            keep = gt_labels > 0
            gt_boxes = gt_boxes[keep]
            gt_labels = gt_labels[keep]

            all_gt[img_id] = (gt_boxes, gt_labels)
            for c in range(1, len(CLASSES)):
                mask = gt_labels == c
                if np.any(mask):
                    gt_boxes_by_class[c][img_id] = gt_boxes[mask]
                else:
                    gt_boxes_by_class[c].setdefault(img_id, np.zeros((0, 4), dtype=np.float32))

            p_boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
            p_labels = out["labels"].detach().cpu().numpy().astype(np.int64)
            p_scores = out["scores"].detach().cpu().numpy().astype(np.float32)

            m = p_scores >= float(pred_min_conf)
            p_boxes, p_labels, p_scores = p_boxes[m], p_labels[m], p_scores[m]
            all_preds[img_id] = (p_boxes, p_labels, p_scores)

            for c in range(1, len(CLASSES)):
                cmask = p_labels == c
                if not np.any(cmask):
                    continue
                for box, score in zip(p_boxes[cmask], p_scores[cmask]):
                    preds_by_class[c].append((img_id, float(score), box.astype(np.float32)))

    ap_list = []
    pr_paths = []

    for c in range(1, len(CLASSES)):
        scores, recall, precision = precision_recall_curve_from_detections(
            preds_by_class[c],
            gt_boxes_by_class[c],
            iou_thres,
        )
        ap = ap_from_pr(recall, precision) if recall.size else float("nan")
        ap_list.append(ap)
        class_name = CLASSES[c]
        p1 = out_path / "pr_curves" / f"PR_{class_name}.png"
        p2 = out_path / "viz2" / f"pr_curve_{class_name}.png"
        plot_pr_curve_per_class(class_name, recall, precision, ap, p1)
        plot_pr_curve_per_class(class_name, recall, precision, ap, p2)
        pr_paths.append(p2)

    mAP50 = float(np.nanmean(np.array(ap_list, dtype=np.float32))) if any(np.isfinite(ap_list)) else 0.0
    plot_ap_bar(class_names, ap_list, out_path / "AP_by_class.png")
    plot_ap_bar(class_names, ap_list, out_path / "viz2" / "ap_per_class.png")

    ap_order = np.argsort(np.where(np.isfinite(ap_list), np.array(ap_list, dtype=np.float32), -1.0))[::-1]
    topk = [i for i in ap_order[:5].tolist() if np.isfinite(ap_list[i])]
    plt.figure(figsize=(8, 6))
    for i in topk:
        c = i + 1
        _, recall, precision = precision_recall_curve_from_detections(preds_by_class[c], gt_boxes_by_class[c], iou_thres)
        if recall.size:
            plt.plot(recall, precision, linewidth=2, label=f"{CLASSES[c]} (AP={ap_list[i]:.3f})")
    plt.title("PR curves (top-5 by AP)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "PR_curve_top5.png", dpi=200)
    plt.close()

    valid_scores = []
    valid_tp = []
    npos_total = 0
    for img_id, (gt_boxes, gt_labels) in all_gt.items():
        npos_total += int(gt_labels.shape[0])
        p_boxes, p_labels, p_scores = all_preds.get(img_id, (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.float32)))
        order = np.argsort(-p_scores)
        p_boxes, p_labels, p_scores = p_boxes[order], p_labels[order], p_scores[order]
        gt_used = np.zeros(gt_labels.shape[0], dtype=bool)
        for pb, pl, ps in zip(p_boxes, p_labels, p_scores):
            valid_scores.append(float(ps))
            candidates = np.where(gt_labels == int(pl))[0]
            if candidates.size == 0:
                valid_tp.append(0.0)
                continue
            ious = box_iou_np(np.asarray([pb], dtype=np.float32), gt_boxes[candidates])[0]
            best_local = int(np.argmax(ious))
            best_gt = int(candidates[best_local])
            if float(ious[best_local]) >= float(iou_thres) and not bool(gt_used[best_gt]):
                gt_used[best_gt] = True
                valid_tp.append(1.0)
            else:
                valid_tp.append(0.0)

    if len(valid_scores):
        order = np.argsort(-np.array(valid_scores, dtype=np.float32))
        scores_sorted = np.array(valid_scores, dtype=np.float32)[order]
        tp_sorted = np.array(valid_tp, dtype=np.float32)[order]
        fp_sorted = 1.0 - tp_sorted
        tp_cum = np.cumsum(tp_sorted)
        fp_cum = np.cumsum(fp_sorted)
        recall_curve = tp_cum / max(float(npos_total), 1.0)
        precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        f1_curve = 2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-9)

        conf_points = np.linspace(0, 1, 200)
        p_at = np.zeros_like(conf_points)
        r_at = np.zeros_like(conf_points)
        f1_at = np.zeros_like(conf_points)
        for i, t in enumerate(conf_points):
            k = int(np.searchsorted(-scores_sorted, -t, side="left"))
            if k <= 0:
                p_at[i] = 0.0
                r_at[i] = 0.0
                f1_at[i] = 0.0
                continue
            tp_k = float(tp_cum[k - 1])
            fp_k = float(fp_cum[k - 1])
            p_at[i] = tp_k / max(tp_k + fp_k, 1e-9)
            r_at[i] = tp_k / max(float(npos_total), 1.0)
            f1_at[i] = 2 * p_at[i] * r_at[i] / max(p_at[i] + r_at[i], 1e-9)

        plot_curve(conf_points, p_at, "Precision vs Confidence", "Confidence", "Precision", out_path / "P_curve.png")
        plot_curve(conf_points, r_at, "Recall vs Confidence", "Confidence", "Recall", out_path / "R_curve.png")
        plot_curve(conf_points, f1_at, "F1 vs Confidence", "Confidence", "F1", out_path / "F1_curve.png")
        plt.figure(figsize=(8, 6))
        plt.plot(conf_points, p_at, label="P")
        plt.plot(conf_points, r_at, label="R")
        plt.plot(conf_points, f1_at, label="F1")
        plt.title("P/R/F1 vs Confidence")
        plt.xlabel("Confidence")
        plt.ylabel("Value")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path / "PRF1_vs_conf.png", dpi=200)
        plt.close()
    else:
        (out_path / "_write_test.txt").write_text("No predictions found.", encoding="utf-8")

    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=np.int64)
    for img_id, (gt_boxes, gt_labels) in all_gt.items():
        p_boxes, p_labels, p_scores = all_preds.get(img_id, (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64), np.zeros((0,), np.float32)))
        m = p_scores >= float(cm_conf_thres)
        p_boxes, p_labels, p_scores = p_boxes[m], p_labels[m], p_scores[m]
        if p_scores.size:
            order = np.argsort(-p_scores)
            p_boxes, p_labels = p_boxes[order], p_labels[order]

        gt_map = gt_labels - 1
        pred_map = p_labels - 1
        matches, unmatched_p, unmatched_g = greedy_match_by_iou(p_boxes, gt_boxes, iou_thres)
        for p_i, g_i, _ in matches:
            t = int(gt_map[g_i]) if 0 <= int(gt_map[g_i]) < n_cls else bg_i
            p = int(pred_map[p_i]) if 0 <= int(pred_map[p_i]) < n_cls else bg_i
            cm[t, p] += 1
        for g_i in unmatched_g:
            t = int(gt_map[g_i]) if 0 <= int(gt_map[g_i]) < n_cls else bg_i
            cm[t, bg_i] += 1
        for p_i in unmatched_p:
            p = int(pred_map[p_i]) if 0 <= int(pred_map[p_i]) < n_cls else bg_i
            cm[bg_i, p] += 1

    names_cm = class_names + ["background"]
    plot_confusion_matrix(cm, names_cm, out_path / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, names_cm, out_path / "confusion_matrix_normalized.png", normalize=True)

    metrics = {
        "weights": str(weights_path),
        "data_root": str(data_root),
        "split": str(split),
        "iou_thres": float(iou_thres),
        "mAP50": float(mAP50),
        "AP50_by_class": {class_names[i]: (None if math.isnan(ap_list[i]) else float(ap_list[i])) for i in range(n_cls)},
        "num_images": int(len(ds)),
        "num_gt": int(sum(all_gt[k][1].shape[0] for k in all_gt)),
    }
    (out_path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "viz2" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"mAP@0.5: {mAP50:.4f}",
        f"IoU threshold: {iou_thres}",
        f"Images: {len(ds)}",
        f"GT boxes: {metrics['num_gt']}",
        "",
        "AP@0.5 by class:",
    ]
    for name in class_names:
        v = metrics["AP50_by_class"][name]
        lines.append(f"  {name}: {v if v is not None else 'nan'}")
    (out_path / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_path / "viz2" / "metrics.txt").write_text("\n".join(lines), encoding="utf-8")

    test_img = out_path / "_mpl_test.png"
    plt.figure(figsize=(2, 2))
    plt.plot([0, 1], [0, 1])
    plt.tight_layout()
    plt.savefig(test_img, dpi=120)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(prog="ssd_result", description="Generate PR curves and evaluation plots for torchvision SSD.")
    p.add_argument("--weights", type=str, default=r"D:\ultralytics_drone\ssd_results\best_ssd.pth")
    p.add_argument("--data-root", type=str, default=r"D:\ultralytics_drone\datasets\VOC2007")
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--out", type=str, default=r"D:\ultralytics_drone\runs\SSD_runs")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--cm-conf", type=float, default=0.25)
    p.add_argument("--pred-min-conf", type=float, default=0.001)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    run_eval(
        weights_path=args.weights,
        data_root=args.data_root,
        split=args.split,
        out_dir=args.out,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        iou_thres=args.iou,
        cm_conf_thres=args.cm_conf,
        pred_min_conf=args.pred_min_conf,
        device=args.device,
    )


if __name__ == "__main__":
    main()

