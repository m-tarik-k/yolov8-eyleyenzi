import os, json, random, shutil, argparse
from PIL import Image
from tqdm import tqdm

# -------------------------------
# ARGUMANLARI OKU
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", required=True, help="Resim ve etiket dosyalarini içeren veri klasörünün yolu")
parser.add_argument("--out", default="dataset-coco", help="cikti klasoru")
parser.add_argument("--train", type=float, default=0.7, help="Eğitim klasörüne gidecek resimlerin orani")
parser.add_argument("--val", type=float, default=0.2, help="Doğrulama klasörüne gidecek resimlerin orani")
parser.add_argument("--test", type=float, default=0.1, help="Test klasörüne gidecek resimlerin orani")
args = parser.parse_args()

DATASET_DIR = args.datapath
OUTPUT_DIR  = args.out
TRAIN_SPLIT = args.train
VAL_SPLIT   = args.val
TEST_SPLIT  = args.test

assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1) < 1e-6, "Oranlarin toplami 1 olmali"

# -------------------------------
# COCO YAPISI
# -------------------------------
def create_coco():
    return {"info": {}, "licenses": [], "categories": [], "images": [], "annotations": []}

# -------------------------------
# YOLO -> COCO DONUSUMU
# -------------------------------
def yolo_to_coco(xc, yc, w, h, iw, ih):
    xc *= iw; yc *= ih; w *= iw; h *= ih
    return [xc - w / 2, yc - h / 2, w, h]

# -------------------------------
# ANA FONKSIYON
# -------------------------------
def main():
    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")
    class_path = os.path.join(DATASET_DIR, "classes.txt")

    if not os.path.exists(class_path):
        print("classes.txt bulunamadi")
        return

    with open(class_path, "r") as f:
        class_list = [c.strip() for c in f.readlines()]

    categories = [{"id": i, "name": c, "supercategory": c} for i, c in enumerate(class_list)]

    imgs = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(imgs)

    n = len(imgs)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    split_data = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train + n_val],
        "test": imgs[n_train + n_val:]
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for s in split_data:
        os.makedirs(os.path.join(OUTPUT_DIR, s), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotations"), exist_ok=True)

    img_id = 0
    ann_id = 0

    for split_name, img_list in split_data.items():
        print(f"\nIsleniyor: {split_name} -> {len(img_list)} resim")

        coco = create_coco()
        coco["categories"] = categories

        for img_name in tqdm(img_list):
            img_path = os.path.join(images_dir, img_name)
            lbl_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

            img = Image.open(img_path)
            w, h = img.size

            coco["images"].append({
                "id": img_id,
                "file_name": img_name,
                "width": w,
                "height": h
            })

            shutil.copy(img_path, os.path.join(OUTPUT_DIR, split_name, img_name))

            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    for line in f:
                        p = line.strip().split()
                        if len(p) != 5: 
                            continue

                        cls = int(p[0])
                        xc, yc, bw, bh = map(float, p[1:])
                        box = yolo_to_coco(xc, yc, bw, bh, w, h)

                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls,
                            "bbox": box,
                            "segmentation": [],
                            "area": box[2] * box[3],
                            "iscrowd": 0
                        })
                        ann_id += 1

            img_id += 1

        out_json = os.path.join(OUTPUT_DIR, "annotations", f"instances_{split_name}.json")
        with open(out_json, "w") as f:
            json.dump(coco, f, indent=4)

    print("\nDonusum tamamlandi")
    print(f"Kayit konumu: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
