''' 
Yüz Tespiti 2
Bu kod, Ultralytics kütüphanesinin YOLO algoritmasini kullanarak bir görseldeki objeleri tespit eder, 
tespit edilen objelerin etrafina bir dörtgen çizer ve tespit edilen objenin sinif ID'sini,sinifini(class ID,class name) ve 
objenin o sinifa dair oldugundan ne kadar emin oldugunu(confidence) gösteren bir yüzde iceren bir yazi ekler.
Sonuç, Haar Cascade algoritmasi ile yüz tespiti yaptigimiz önceki koda göre çok daha isabetli olacaktir.
''' 

from ultralytics import YOLO # Ultralytics kütüphanesinin YOLO algoraritması
import cv2 # Görüntü işleme
from pathlib import Path # Dosya yolu işlemleri

# Görsel dosya yolunu belirle
root = Path.cwd()
image = root / 'Projects' / 'YuzTespiti_2' / 'Faces.webp'

# YOLO modelini yükle (önceden eğitilmiş model)
model = YOLO('yolov8s.pt') 
image = cv2.imread(image)

# Görüntüdeki objeleri modele tespit ettir
results = model(image) # "results", tespit edilen objeler hakkında bilgi içerir

# Tespit edilen objelerin etrafına dikdörtgen çiz ve yazı ekle
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Dörgenin koordinatları
        confidence = box.conf[0]  # "Eminiyet" değeri (Confidence)
        class_id = int(box.cls[0])  # Sınıf ID'si (Class ID)
        label = model.names[class_id]  # Sınıf adı (Class name)

        # Dikdörtgen çiz
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Etiket ekle
        cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Sonucu göster
cv2.imshow('Yuz Tespiti', image)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

