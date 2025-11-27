# Dataseti eğitim (train) ve sinama (validation) klasörlerine bölme

from pathlib import Path
import random
import os
import sys
import shutil
import argparse


# Kullanici girdisi argümanlarini tanimla ve ayriştir
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Resim ve etiket dosyalarini içeren veri klasörünün yolu',
                    required=True)
parser.add_argument('--train_pct', help='Eğitim klasörüne gidecek resimlerin orani; \
                    kalanlar doğrulama klasörüne gider (örnek: ".8")',
                    default=.8)

args = parser.parse_args()

data_path = args.datapath
train_percent = float(args.train_pct)

# Geçerli girişleri kontrol et
if not os.path.isdir(data_path):
   print('---datapath tarafindan belirtilen klasör bulunamadi. Yolun doğru olduğunu doğrulayin (Windows için çift ters eğik çizgi \\ kullanin) ve tekrar deneyin.')
   sys.exit(0)
if train_percent < .01 or train_percent > 0.99:
   print('train_pct için geçersiz değer. Lütfen .01 ile .99 arasinda bir sayi girin.')
   sys.exit(0)
val_percent = 1 - train_percent

# Girdi veri seti yolu
input_image_path = os.path.join(data_path,'images')
input_label_path = os.path.join(data_path,'labels')

# Resim ve etiket klasör yollarini tanimla
cwd = os.getcwd()
train_img_path = os.path.join(cwd,'data/train/images')
train_txt_path = os.path.join(cwd,'data/train/labels')
val_img_path = os.path.join(cwd,'data/validation/images')
val_txt_path = os.path.join(cwd,'data/validation/labels')

# Klasörleri oluştur (mevcut değillerse)
for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
   if not os.path.exists(dir_path):
      os.makedirs(dir_path)
      print(f'{dir_path} konumunda klasör oluşturuldu.')

# Tüm resim ve etiket dosyalarinin listesini al
img_file_list = [path for path in Path(input_image_path).rglob('*')]
txt_file_list = [path for path in Path(input_label_path).rglob('*')]

print(f'Görsel dosyasi sayisi: {len(img_file_list)}')
print(f'Etiket dosyasi sayisi: {len(txt_file_list)}')

# Her klasöre taşinacak dosya sayisini belirle
file_num = len(img_file_list)
train_num = int(file_num*train_percent)
val_num = file_num - train_num
print('Eğitim klasörüne taşinacak resim sayisi: %d' % train_num)
print('Doğrulama klasörüne taşinacak resim sayisi: %d' % val_num)

# Dosyalari rastgele seçip train veya val klasörlerine kopyala
for i, set_num in enumerate([train_num, val_num]):
  for ii in range(set_num):
    img_path = random.choice(img_file_list)
    img_fn = img_path.name
    base_fn = img_path.stem
    txt_fn = base_fn + '.txt'
    txt_path = os.path.join(input_label_path,txt_fn)

    if i == 0:  # İlk grup eğitim klasörüne kopyalanir
      new_img_path, new_txt_path = train_img_path, train_txt_path
    elif i == 1:  # İkinci grup doğrulama klasörüne kopyalanir
      new_img_path, new_txt_path = val_img_path, val_txt_path

    shutil.copy(img_path, os.path.join(new_img_path,img_fn))
    # os.rename(img_path, os.path.join(new_img_path,img_fn))  # Kopyalamak yerine taşimak isterseniz kullanabilirsiniz
    if os.path.exists(txt_path):  # Eğer txt dosyasi yoksa bu arka plan resmidir, txt dosyasi atlanir
      shutil.copy(txt_path,os.path.join(new_txt_path,txt_fn))
      # os.rename(txt_path,os.path.join(new_txt_path,txt_fn))  # Kopyalamak yerine taşimak isterseniz kullanabilirsiniz

    img_file_list.remove(img_path)
