import os

# ===================== 你的路径 =====================
TRAIN_LABELS = r"E:\2026\ultralytics_drone\VisDrone2019-DET-train\labels"
VAL_LABELS = r"E:\2026\ultralytics_drone\VisDrone2019-DET-val\labels"


# ====================================================

def fix_labels(label_dir):
    for txt in os.listdir(label_dir):
        if not txt.endswith(".txt"):
            continue
        path = os.path.join(label_dir, txt)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            # 关键：VisDrone 1→0, 2→1 ... 10→9
            if cls_id == 0:
                continue
            new_cls = cls_id - 1
            if 0 <= new_cls <= 9:
                parts[0] = str(new_cls)
                new_lines.append(" ".join(parts) + "\n")

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    print(f"✅ 标签修复完成：{label_dir}")


fix_labels(TRAIN_LABELS)
fix_labels(VAL_LABELS)