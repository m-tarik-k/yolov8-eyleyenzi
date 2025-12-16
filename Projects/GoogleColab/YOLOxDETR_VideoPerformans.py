import cv2
import time
import argparse
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rfdetr.detr import RFDETRSmall, RFDETRMedium, RFDETRLarge

# ============================================================
# ======================= ARGÜMANLAR =========================
# ============================================================

parser = argparse.ArgumentParser(
    description="YOLO ve RF-DETR modellerini video üzerinde karşilaştir"
)

parser.add_argument("--video", type=str, required=True,
                    help="Giriş video yolu")

parser.add_argument("--yolo_model", type=str, required=True,
                    help="YOLO model dosyasi (.pt)")

parser.add_argument("--detr_model", type=str, required=True,
                    help="RF-DETR model dosyasi (.pth)")

parser.add_argument("--detr_size", type=str, default="small",
                    choices=["small", "medium", "large"],
                    help="RF-DETR model boyutu")

parser.add_argument("--output", type=str, default="comparison_output.mp4",
                    help="Çikiş video dosyasi")

parser.add_argument("--device", type=str, default="cuda",
                    help="cuda veya cpu")

parser.add_argument("--start_time", type=float, default=0.0,
                    help="Video başlangiç zamani (saniye)")

parser.add_argument("--end_time", type=float, default=10.0,
                    help="Video bitiş zamani (saniye)")

parser.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold")

args = parser.parse_args()

# ============================================================
# ======================== SABİTLER ==========================
# ============================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ============================================================
# ======================= MODELLER ===========================
# ============================================================

print("[INFO] YOLO yükleniyor...")
yolo_model = YOLO(args.yolo_model)

print("[INFO] RF-DETR yükleniyor...")
if args.detr_size == "small":
    rfdetr_model = RFDETRSmall(pretrain_weights=args.detr_model, device=args.device)
elif args.detr_size == "medium":
    rfdetr_model = RFDETRMedium(pretrain_weights=args.detr_model, device=args.device)
else:
    rfdetr_model = RFDETRLarge(pretrain_weights=args.detr_model, device=args.device)

# Sinif isimlerini YOLO'dan al
CLASS_NAMES = yolo_model.names
rfdetr_model.custom_classes = list(CLASS_NAMES.values())

# ============================================================
# ======================= VIDEO SETUP ========================
# ============================================================

cap = cv2.VideoCapture(args.video)

fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

start_frame = int(args.start_time * fps)
end_frame   = int(args.end_time * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

out = cv2.VideoWriter(
    args.output,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W * 2, H)
)

# ============================================================
# ======================= METRİKLER ==========================
# ============================================================

yolo_times = []
rfdetr_times = []
yolo_det_count = []
rfdetr_det_count = []

# ============================================================
# ===================== ÇİZİM FONKSİYONU =====================
# ============================================================

def draw_boxes(img, boxes, labels, confs, color):
    """
    Bounding box ve etiketleri ekrana çizer
    """
    for (x1, y1, x2, y2), cls, conf in zip(boxes, labels, confs):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{cls} {conf:.2f}"
        cv2.putText(
            img, text, (x1, max(20, y1 - 5)),
            FONT, 0.5, color, 2
        )

# ============================================================
# ======================== ANA DÖNGÜ =========================
# ============================================================

frame_idx = start_frame

while cap.isOpened() and frame_idx <= end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------
    t0 = time.time()
    yolo_res = yolo_model(frame, verbose=False)[0]
    yolo_times.append(time.time() - t0)

    y_boxes, y_labels, y_confs = [], [], []

    if yolo_res.boxes is not None:
        for b in yolo_res.boxes:
            if b.conf[0] < args.conf:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            y_boxes.append((x1, y1, x2, y2))
            y_labels.append(CLASS_NAMES[int(b.cls[0])])
            y_confs.append(float(b.conf[0]))

    yolo_det_count.append(len(y_boxes))
    yolo_img = frame.copy()
    draw_boxes(yolo_img, y_boxes, y_labels, y_confs, (0, 255, 0))

    # --------------------------------------------------------
    # RF-DETR inference
    # --------------------------------------------------------
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    t1 = time.time()
    r_res = rfdetr_model.predict(pil_img)
    rfdetr_times.append(time.time() - t1)

    r_boxes, r_labels, r_confs = [], [], []

    for box, conf, cls in zip(r_res.xyxy, r_res.confidence, r_res.class_id):
        if conf < args.conf:
            continue
        x1, y1, x2, y2 = map(int, box)
        r_boxes.append((x1, y1, x2, y2))
        r_labels.append(CLASS_NAMES[int(cls)])
        r_confs.append(float(conf))

    rfdetr_det_count.append(len(r_boxes))
    rfdetr_img = frame.copy()
    draw_boxes(rfdetr_img, r_boxes, r_labels, r_confs, (255, 0, 0))

    # --------------------------------------------------------
    # FPS overlay
    # --------------------------------------------------------
    yolo_fps = 1 / np.mean(yolo_times[-10:]) if len(yolo_times) >= 10 else 0
    rfdetr_fps = 1 / np.mean(rfdetr_times[-10:]) if len(rfdetr_times) >= 10 else 0

    cv2.putText(yolo_img, f"YOLO | FPS: {yolo_fps:.1f}",
                (10, 30), FONT, 1, (0, 255, 0), 2)

    cv2.putText(rfdetr_img, f"RF-DETR | FPS: {rfdetr_fps:.1f}",
                (10, 30), FONT, 1, (255, 0, 0), 2)

    # --------------------------------------------------------
    # Yan yana birleştir ve videoya yaz
    # --------------------------------------------------------
    combined = np.hstack([yolo_img, rfdetr_img])
    out.write(combined)

    frame_idx += 1

# ============================================================
# ========================= ÖZET =============================
# ============================================================

cap.release()
out.release()

print("\n========= PERFORMANS ÖZETİ =========")
print(f"Zaman araliği: {args.start_time}s → {args.end_time}s")

print("\nYOLO:")
print(f"  Ortalama FPS: {1 / np.mean(yolo_times):.2f}")
print(f"  Ortalama tespit/frame: {np.mean(yolo_det_count):.2f}")

print("\nRF-DETR:")
print(f"  Ortalama FPS: {1 / np.mean(rfdetr_times):.2f}")
print(f"  Ortalama tespit/frame: {np.mean(rfdetr_det_count):.2f}")

print(f"\nKarşilaştirma videosu kaydedildi → {args.output}")
