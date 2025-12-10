import os
import time
import yaml
import tempfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from rfdetr.detr import RFDETRSmall

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DEVICE = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="/content/dataset1", help="Dataset klasoru pathi")
parser.add_argument("--yolo_dir", type=str, default="/content/yolo.pt", help="YOLO modeli pathi")
parser.add_argument("--detr_dir", type=str, default="/content/detr.pth", help="DETR modeli pathi")
parser.add_argument("--output_dir", type=str, default="/content/sonuc", help="Karsilastirma sonuclarinin cikti klasoru")

DATA_DIR = parser.parse_args().dataset_dir
IMG_DIR = os.path.join(DATA_DIR, "images/")
LBL_DIR = os.path.join(DATA_DIR, "labels/")
CLS_DIR = os.path.join(DATA_DIR, "classes.txt")

YOLO_MODEL_PATH = parser.parse_args().yolo_dir
RFDETR_MODEL_PATH = parser.parse_args().detr_dir

CLASS_NAMES = []
NUM_CLASSES = 0

OUTPUT_DIR = parser.parse_args().output_dir
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "results.txt")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "results.png")

# ------------------------------------------------------------
# Fetch class names and number of classes
# ------------------------------------------------------------

with open(CLS_DIR, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines() if line.strip()]
NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------------------------------------
# Load YOLO txt annotation and convert to xyxy
# ------------------------------------------------------------
def load_yolo_labels(path, img_w, img_h):
    boxes = []
    with open(path, "r") as f:
        for line in f.readlines():
            cls, xc, yc, w, h = map(float, line.strip().split())

            xc *= img_w
            yc *= img_h
            w  *= img_w
            h  *= img_h

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            boxes.append((int(cls), x1, y1, x2, y2))
    return boxes

# ------------------------------------------------------------
# Calc IoU
# ------------------------------------------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter + 1e-9
    return inter / union

# ------------------------------------------------------------
# Calc AP 
# ------------------------------------------------------------
def compute_ap(preds, gts, iou_thresh):
    preds = sorted(preds, key=lambda x: -x[2])

    TP = 0
    FP = 0
    matched = set()

    for p in preds:
        p_img, p_cls, p_conf, px1, py1, px2, py2 = p
        best_iou = 0
        best_gt = None

        for gi, g in enumerate(gts):
            g_img, g_cls, gx1, gy1, gx2, gy2 = g
            if g_img != p_img or g_cls != p_cls:
                continue

            i = iou((px1,py1,px2,py2), (gx1,gy1,gx2,gy2))
            if i > best_iou:
                best_iou = i
                best_gt = gi

        if best_iou >= iou_thresh and best_gt not in matched:
            TP += 1
            matched.add(best_gt)
        else:
            FP += 1

    FN = len(gts) - len(matched)

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)

    return precision * recall

# ============================================================
# =================== RF-DETR EVALUATION =====================
# ============================================================
print("Evaluating RF-DETR...")

rfdetr_model = RFDETRSmall(pretrain_weights=RFDETR_MODEL_PATH, device=DEVICE)
rfdetr_model.custom_classes = CLASS_NAMES

preds = []
gts = []

speed_pre = []
speed_inf = []
speed_post = []

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    base = os.path.splitext(img_name)[0]
    txt_path = os.path.join(LBL_DIR, base + ".txt")
    img_path = os.path.join(IMG_DIR, img_name)

    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    if os.path.exists(txt_path):
        gts_list = load_yolo_labels(txt_path, W, H)
        for cls, x1,y1,x2,y2 in gts_list:
            gts.append((base, cls, x1, y1, x2, y2))

    t0 = time.time()
    img_pil = img.copy()
    speed_pre.append((time.time() - t0) * 1000)

    t1 = time.time()
    results = rfdetr_model.predict(img_pil)
    speed_inf.append((time.time() - t1) * 1000)

    t2 = time.time()
    for box, conf, cls in zip(results.xyxy, results.confidence, results.class_id):
        x1,y1,x2,y2 = map(float, box)
        preds.append((base, int(cls), float(conf), x1,y1,x2,y2))
    speed_post.append((time.time() - t2) * 1000)

ap50_list = []
ap5095_list = []

for cls in range(NUM_CLASSES):
    preds_c = [p for p in preds if p[1] == cls]
    gts_c = [g for g in gts if g[1] == cls]

    ap50 = compute_ap(preds_c, gts_c, 0.5)
    ap50_list.append(ap50)

    aps = []
    for t in np.linspace(0.5, 0.95, 10):
        aps.append(compute_ap(preds_c, gts_c, t))
    ap5095_list.append(sum(aps) / len(aps))

rfdetr_map50 = sum(ap50_list) / NUM_CLASSES
rfdetr_map5095 = sum(ap5095_list) / NUM_CLASSES

# ============================================================
# ===================== YOLO EVALUATION ======================
# ============================================================
print("Evaluating YOLO...")

dataset_yaml = {
    "path": os.path.abspath("dataset1"),
    "train": "images",
    "val": "images",
    "test": "images",
    "names": CLASS_NAMES,
}

with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
    yaml.dump(dataset_yaml, tmp)
    yaml_path = tmp.name

yolo_model = YOLO(YOLO_MODEL_PATH)

start_time = time.time()
yolo_results = yolo_results = yolo_model.val(
    data=yaml_path,
    verbose=False,
    save=False,
    save_json=False,
    project=".",
    name="temp",
    exist_ok=True
)
eval_time = time.time() - start_time

yolo_map50 = yolo_results.box.map50
yolo_map5095 = yolo_results.box.map
yolo_per_class_ap = yolo_results.box.maps
yolo_speed = yolo_results.speed

os.remove(yaml_path)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ====================== SAVE COMPARISON =====================
# ============================================================
with open(OUTPUT_TXT, "w") as f:
    f.write("========= MODEL COMPARISON =========\n\n")

    # ==========================
    # RF-DETR RESULTS
    # ==========================
    f.write("=== RF-DETR RESULTS ===\n\n")
    f.write(f"mAP50: {rfdetr_map50:.6f}\n")
    f.write(f"mAP50-95: {rfdetr_map5095:.6f}\n\n")

    f.write("Per-class AP:\n")
    for i, cls in enumerate(CLASS_NAMES):
        f.write(f"  {cls}: {ap5095_list[i]:.6f}\n")

    rfdetr_pre  = sum(speed_pre) / len(speed_pre)
    rfdetr_inf  = sum(speed_inf) / len(speed_inf)
    rfdetr_post = sum(speed_post) / len(speed_post)
    rfdetr_total = rfdetr_pre + rfdetr_inf + rfdetr_post

    f.write("\nSpeed (ms):\n")
    f.write(f"  preprocess: {rfdetr_pre:.3f}\n")
    f.write(f"  inference: {rfdetr_inf:.3f}\n")
    f.write(f"  postprocess: {rfdetr_post:.3f}\n")
    f.write(f"  total: {rfdetr_total:.3f}\n")

    # ==========================
    # YOLO RESULTS
    # ==========================
    f.write("\n\n=== YOLO RESULTS ===\n\n")
    f.write(f"mAP50: {yolo_map50:.6f}\n")
    f.write(f"mAP50-95: {yolo_map5095:.6f}\n\n")

    f.write("Per-class AP:\n")
    for i, ap in enumerate(yolo_per_class_ap):
        class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
        f.write(f"  {class_name}: {ap:.6f}\n")

    yolo_total = (
        yolo_speed["preprocess"] +
        yolo_speed["inference"] +
        yolo_speed["postprocess"]
    )

    f.write("\nSpeed (ms):\n")
    f.write(f"  preprocess: {yolo_speed['preprocess']:.3f}\n")
    f.write(f"  inference: {yolo_speed['inference']:.3f}\n")
    f.write(f"  postprocess: {yolo_speed['postprocess']:.3f}\n")
    f.write(f"  total: {yolo_total:.3f}\n")

    # ==========================
    # AUTO COMPARISON
    # ==========================
    f.write("\n\n========= AUTO COMPARISON =========\n\n")

    acc_winner = "RF-DETR" if rfdetr_map5095 > yolo_map5095 else "YOLO"
    speed_winner = "RF-DETR" if rfdetr_total < yolo_total else "YOLO"

    f.write(f"Accuracy Winner (mAP50-95): {acc_winner}\n")
    f.write(f"Speed Winner (Total Time): {speed_winner}\n\n")

    if acc_winner == speed_winner:
        overall = acc_winner
    else:
        if rfdetr_map5095 > yolo_map5095 and rfdetr_total < yolo_total:
            overall = "RF-DETR (Best Accuracy + Speed)"
        elif yolo_map5095 > rfdetr_map5095 and yolo_total < rfdetr_total:
            overall = "YOLO (Best Accuracy + Speed)"
        else:
            overall = "Trade-off (One is faster, one is more accurate)"

    f.write(f"- OVERALL WINNER: {overall}\n")

fig, axes = plt.subplots(3, 1, figsize=(10, 14))

models = ["RF-DETR", "YOLO"]
x_models = np.arange(len(models))

# ------------------------------------------------------------
#  mAP COMPARISON
# ------------------------------------------------------------
map50_vals = [rfdetr_map50, yolo_map50]
map5095_vals = [rfdetr_map5095, yolo_map5095]

axes[0].bar(x_models - 0.15, map50_vals, width=0.3, label="mAP50")
axes[0].bar(x_models + 0.15, map5095_vals, width=0.3, label="mAP50-95")
axes[0].set_xticks(x_models)
axes[0].set_xticklabels(models)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("mAP Comparison")
axes[0].legend()
axes[0].grid(True)

# ------------------------------------------------------------
#  SPEED COMPARISON
# ------------------------------------------------------------
speed_labels = ["Preprocess", "Inference", "Postprocess", "Total"]
x_speed = np.arange(len(speed_labels))

rfdetr_speeds = [
    rfdetr_pre,
    rfdetr_inf,
    rfdetr_post,
    rfdetr_total
]

yolo_speeds = [
    yolo_speed["preprocess"],
    yolo_speed["inference"],
    yolo_speed["postprocess"],
    yolo_total
]

axes[1].bar(x_speed - 0.15, rfdetr_speeds, width=0.3, label="RF-DETR")
axes[1].bar(x_speed + 0.15, yolo_speeds, width=0.3, label="YOLO")
axes[1].set_xticks(x_speed)
axes[1].set_xticklabels(speed_labels)
axes[1].set_ylabel("Milliseconds")
axes[1].set_title("Speed Comparison")
axes[1].legend()
axes[1].grid(True)

# ------------------------------------------------------------
#  PER-CLASS AP COMPARISON
# ------------------------------------------------------------
x_cls = np.arange(NUM_CLASSES)

axes[2].bar(x_cls - 0.15, ap5095_list, width=0.3, label="RF-DETR")
axes[2].bar(x_cls + 0.15, yolo_per_class_ap, width=0.3, label="YOLO")
axes[2].set_xticks(x_cls)
axes[2].set_xticklabels(CLASS_NAMES, rotation=45)
axes[2].set_ylabel("AP (mAP50-95)")
axes[2].set_title("Per-Class AP Comparison")
axes[2].legend()
axes[2].grid(True)

# ------------------------------------------------------------
# FINAL EXPORT
# ------------------------------------------------------------
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=250)
plt.close()

os.system("clear")
print(f"RF-DETR vs YOLO comparison Graphs .png saved → {OUTPUT_PNG}")
print(f"RF-DETR vs YOLO comparison .txt saved → {OUTPUT_TXT}")
