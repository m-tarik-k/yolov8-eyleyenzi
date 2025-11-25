''' 
Yüz Tespiti
Bu kod, OpenCV kütüphanesini kullanarak bir görüntüdeki yüzleri tespit eder ve tespit edilen yüzlerin etrafina dikdörtgen çizer.
Eger dikkat ederseniz kullandigimiz "Cascade" siniflandiricisi(bulunmak istenen obejnin kalibi) sabittir ve önden(frontal) olmayan  
yüzleri tespit ederken zorlanacaktir.
''' 

import cv2 # Görüntü işleme

# Yüz tanıma için Haar Cascade sınıflandırıcısını yükle
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # OpenCV'nin önceden oluşturulmuş Haar Cascade modeli
face_cascade = cv2.CascadeClassifier(cascade_path)

image = cv2.imread("//home//tariktaratas//Desktop//Projects//Python//ComputerVision//Projects//YuzTespiti_1//Faces.webp")

# Görüntüyü gri tonlamaya çevir
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Yüzleri tespit et
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
''' 
ScaleFactor: Görüntüyü ne kadar küçülteceğimiz(hassasiyet~), 
minNeighbors: Tespit için gereken minimum komşu sayisi(kesinlik~), 
minSize: Tespit edilecek yüzlerin minimum boyutu 
'''


# Tespit edilen yüzlerin etrafına dikdörtgen çiz
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Sonucu göster
cv2.imshow('Yuz Tespiti', image)
cv2.waitKey(0) # Herhangi bir tuşa basılana kadar bekle
cv2.destroyAllWindows() # Tüm pencereleri kapat

