# Python fonksiyonu data.yaml config dosyasini otomatik olarak olusturur
# 1. "classes.txt" dosyasini okuyarak sinif isimlerini alir
# 2. Klasor yollarini, sinif sayisini ve sinif isimlerini iceren data listesini olusturur
# 3. Veriyi YAML formatinda data.yaml dosyasina yazar

import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):

  # Sinif isimlerini almak icin classes.txt dosyasini oku
  if not os.path.exists(path_to_classes_txt):
    print(f'classes.txt dosyasi bulunamadi! Lutfen bir classes.txt labelmap olusturun ve {path_to_classes_txt} konumuna tasiyin')
    return
  with open(path_to_classes_txt, 'r') as f:
    classes = []
    for line in f.readlines():
      if len(line.strip()) == 0: continue
      classes.append(line.strip())
  number_of_classes = len(classes)

  # Data sozlugunu olustur
  data = {
      'path': '/content/data',
      'train': 'train/images',
      'val': 'validation/images',
      'nc': number_of_classes,
      'names': classes
  }

  # Veriyi YAML dosyasina yaz
  with open(path_to_data_yaml, 'w') as f:
    yaml.dump(data, f, sort_keys=False)
  print(f'Config dosyasi olusturuldu: {path_to_data_yaml}')

  return

# classes.txt dosya yolunu tanimla ve fonksiyonu calistir
path_to_classes_txt = '/content/custom_data/classes.txt'
path_to_data_yaml = '/content/data.yaml'

create_data_yaml(path_to_classes_txt, path_to_data_yaml)

print('\nDosya icerigi:\n')
