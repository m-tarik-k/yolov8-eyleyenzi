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
# SINIF ISIMLERI
# ------------------------------------------------------------
with open(CLS_DIR) as f:
    CLASS_NAMES = [l.strip() for l in f if l.strip()]
NUM_CLASSES = len(CLASS_NAMES)
BG_IDX = NUM_CLASSES  # background index

# ------------------------------------------------------------
# YARDIMCI FONKSIYONLAR
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
    raise ValueError("Bilinmeyen RF-DETR boyutu")

# ============================================================
# RF-DETR DEGERLENDIRME + CONFUSION MATRIX
# ============================================================
print("RF-DETR degerlendiriliyor...")

rfdetr = load_rfdetr_model(RFDETR_MODEL_PATH, DETR_SIZE, DEVICE)
rfdetr.custom_classes = CLASS_NAMES

cm_rfdetr = np.zeros((NUM_CLASSES+1, NUM_CLASSES+1))

speed_pre, speed_inf, speed_post = [], [], []

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".jpg",".png",".jpeg")):
        continue

    base = os.path.splitext(img_name)[0]
    img = Image.open(os.path.join(IMG_DIR, img_name)).convert("RGB")
    W, H = img.size

    txt_path = os.path.join(LBL_DIR, base + ".txt")
    gt_boxes = load_yolo_labels(txt_path, W, H) if os.path.exists(txt_path) else []

    t0 = time.time()
    speed_pre.append((time.time() - t0) * 1000)

    t1 = time.time()
    res = rfdetr.predict(img)
    speed_inf.append((time.time() - t1) * 1000)

    t2 = time.time()
    used_gt = set()

    for box, cls in zip(res.xyxy, res.class_id):
        px1,py1,px2,py2 = map(float, box)
        best_iou, best_gt = 0, None

        for i,(gcls,gx1,gy1,gx2,gy2) in enumerate(gt_boxes):
            if i in used_gt:
                continue
            v = iou((px1,py1,px2,py2),(gx1,gy1,gx2,gy2))
            if v > best_iou:
                best_iou, best_gt = v, i

        if best_iou >= 0.5:
            cm_rfdetr[gt_boxes[best_gt][0], int(cls)] += 1
            used_gt.add(best_gt)
        else:
            cm_rfdetr[BG_IDX, int(cls)] += 1

    for i,(gcls, *_ ) in enumerate(gt_boxes):
        if i not in used_gt:
            cm_rfdetr[gcls, BG_IDX] += 1

    speed_post.append((time.time() - t2) * 1000)

cm_rfdetr /= (cm_rfdetr.sum(axis=1, keepdims=True) + 1e-9)

# ============================================================
# YOLO DEGERLENDIRME + CONFUSION MATRIX
# ============================================================
print("YOLO degerlendiriliyor...")

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
cm_yolo /= (cm_yolo.sum(axis=1, keepdims=True) + 1e-9)

yolo_speed = yolo_results.speed
yolo_map50 = yolo_results.box.map50
yolo_map5095 = yolo_results.box.map

os.remove(yaml_path)

# ============================================================
# CONFUSION MATRIX CIZIMI (YESIL / BEYAZ)
# ============================================================
labels = CLASS_NAMES + ["background"]

fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="white")

def plot_cm(ax, cm, title):
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(im, ax=ax)

plot_cm(axes[0], cm_rfdetr, "RF-DETR Normalize Confusion Matrix")
plot_cm(axes[1], cm_yolo, "YOLO Normalize Confusion Matrix")

plt.tight_layout()
plt.savefig(OUTPUT_CM_PNG, dpi=250)
plt.close()

# ============================================================
# SONUC GRAFIKLERI (results.png) — ARTIK BOS DEGIL
# ============================================================
rfdetr_total = np.mean(speed_pre) + np.mean(speed_inf) + np.mean(speed_post)
yolo_total = yolo_speed["preprocess"] + yolo_speed["inference"] + yolo_speed["postprocess"]

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].bar(["RF-DETR", "YOLO"], [rfdetr_total, yolo_total])
ax[0].set_title("Toplam Sure (ms)")
ax[0].set_ylabel("ms")

ax[1].bar(["RF-DETR", "YOLO"], [0, yolo_map5095])
ax[1].set_title("mAP50-95")
ax[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=250)
plt.close()

print(f"✓ Sonuc grafikleri kaydedildi → {OUTPUT_PNG}")
print(f"✓ Confusion matrix kaydedildi → {OUTPUT_CM_PNG}")
print("BITTI.")
