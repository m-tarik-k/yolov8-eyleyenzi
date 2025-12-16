import os
import time
import yaml
import tempfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from rfdetr.detr import RFDETRSmall, RFDETRMedium, RFDETRLarge

# ------------------------------------------------------------
# AYARLAR
# ------------------------------------------------------------
DEVICE = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="/content/dataset1")
parser.add_argument("--yolo_dir", type=str, default="/content/yolo.pt")
parser.add_argument("--detr_dir", type=str, default="/content/detr.pth")
parser.add_argument("--output_dir", type=str, default="/content/sonuc")
parser.add_argument("--detr_size", type=str, default="small", choices=["small","medium","large"])

args = parser.parse_args()

DATA_DIR = args.dataset_dir
IMG_DIR = os.path.join(DATA_DIR, "images")
LBL_DIR = os.path.join(DATA_DIR, "labels")
CLS_DIR = os.path.join(DATA_DIR, "classes.txt")

YOLO_MODEL_PATH = args.yolo_dir
RFDETR_MODEL_PATH = args.detr_dir
DETR_SIZE = args.detr_size

OUTPUT_DIR = args.output_dir
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "results.txt")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "results.png")
OUTPUT_CM_PNG = os.path.join(OUTPUT_DIR, "confusion_matrices.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# CLASS NAMES
# ------------------------------------------------------------
with open(CLS_DIR) as f:
    CLASS_NAMES = [l.strip() for l in f if l.strip()]
NUM_CLASSES = len(CLASS_NAMES)
BG_IDX = NUM_CLASSES  # background index

# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------
def load_yolo_labels(path, w, h):
    boxes = []
    with open(path) as f:
        for l in f:
            cls, xc, yc, bw, bh = map(float, l.split())
            xc *= w; yc *= h; bw *= w; bh *= h
            x1 = xc - bw/2; y1 = yc - bh/2
            x2 = xc + bw/2; y2 = yc + bh/2
            boxes.append((int(cls), x1, y1, x2, y2))
    return boxes

def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-9)

def load_rfdetr_model(path, size, device):
    if size == "small":
        return RFDETRSmall(pretrain_weights=path, device=device)
    if size == "medium":
        return RFDETRMedium(pretrain_weights=path, device=device)
    if size == "large":
        return RFDETRLarge(pretrain_weights=path, device=device)
    raise ValueError("Unknown RF-DETR size")

# ------------------------------------------------------------
# RF-DETR EVALUATION + CONFUSION MATRIX
# ------------------------------------------------------------
print("Evaluating RF-DETR...")

rfdetr = load_rfdetr_model(RFDETR_MODEL_PATH, DETR_SIZE, DEVICE)
rfdetr.custom_classes = CLASS_NAMES

cm_rfdetr = np.zeros((NUM_CLASSES+1, NUM_CLASSES+1))
preds, gts = [], []

speed_pre, speed_inf, speed_post = [], [], []

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".jpg",".png",".jpeg")):
        continue

    base = os.path.splitext(img_name)[0]
    img_path = os.path.join(IMG_DIR, img_name)
    txt_path = os.path.join(LBL_DIR, base + ".txt")

    img = Image.open(img_path).convert("RGB")
    W,H = img.size

    gt_boxes = []
    if os.path.exists(txt_path):
        gt_boxes = load_yolo_labels(txt_path, W, H)
        for g in gt_boxes:
            gts.append((base, *g))

    t0 = time.time()
    speed_pre.append((time.time()-t0)*1000)

    t1 = time.time()
    res = rfdetr.predict(img)
    speed_inf.append((time.time()-t1)*1000)

    t2 = time.time()
    used_gt = set()

    for box, conf, cls in zip(res.xyxy, res.confidence, res.class_id):
        px1,py1,px2,py2 = map(float, box)
        best_iou, best_gt = 0, None

        for i,(gcls,gx1,gy1,gx2,gy2) in enumerate(gt_boxes):
            if i in used_gt:
                continue
            v = iou((px1,py1,px2,py2),(gx1,gy1,gx2,gy2))
            if v > best_iou:
                best_iou, best_gt = v, i

        if best_iou >= 0.5:
            gt_cls = gt_boxes[best_gt][0]
            cm_rfdetr[gt_cls, int(cls)] += 1
            used_gt.add(best_gt)
        else:
            cm_rfdetr[BG_IDX, int(cls)] += 1

    for i,(gcls, *_ ) in enumerate(gt_boxes):
        if i not in used_gt:
            cm_rfdetr[gcls, BG_IDX] += 1

    speed_post.append((time.time()-t2)*1000)

# Normalize
cm_rfdetr = cm_rfdetr / (cm_rfdetr.sum(axis=1, keepdims=True) + 1e-9)

# ------------------------------------------------------------
# YOLO EVALUATION + CONFUSION MATRIX
# ------------------------------------------------------------
print("Evaluating YOLO...")

dataset_yaml = {
    "path": DATA_DIR,
    "train": "images",
    "val": "images",
    "names": CLASS_NAMES
}

with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
    yaml.dump(dataset_yaml, f)
    yaml_path = f.name

yolo = YOLO(YOLO_MODEL_PATH)
yolo_results = yolo.val(data=yaml_path, verbose=False)

cm_yolo = yolo_results.confusion_matrix.matrix
cm_yolo = cm_yolo / (cm_yolo.sum(axis=1, keepdims=True) + 1e-9)

os.remove(yaml_path)

# ------------------------------------------------------------
# CONFUSION MATRIX PLOT
# ------------------------------------------------------------
fig, axes = plt.subplots(1,2, figsize=(18,8), facecolor="#0f0f0f")

labels = CLASS_NAMES + ["background"]

def plot_cm(ax, cm, title):
    im = ax.imshow(cm, cmap="magma")
    ax.set_title(title, fontsize=14, color="white")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", color="white")
    ax.set_yticklabels(labels, color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Ground Truth", color="white")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,f"{cm[i,j]:.2f}",ha="center",va="center",
                    color="white" if cm[i,j]>0.5 else "black", fontsize=8)

    ax.figure.colorbar(im, ax=ax)

plot_cm(axes[0], cm_rfdetr, "RF-DETR Normalized Confusion Matrix")
plot_cm(axes[1], cm_yolo, "YOLO Normalized Confusion Matrix")

plt.tight_layout()
plt.savefig(OUTPUT_CM_PNG, dpi=250)
plt.close()

# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------
print(f"Confusion matrices saved → {OUTPUT_CM_PNG}")
print("DONE.")
