import pandas as pd
import os
from PIL import Image


CHICKEN_DIR = '/home/jiaendong/rabbitsRatsChicken/' + 'train/' + 'chicken/'
chicken_name = os.listdir(CHICKEN_DIR)
chicken_name.sort(key=lambda x: int(x[4:-4]))
chicken_info = {'path': [], 'species': [], 'class': []}

for file in chicken_name:
    try:
        a = Image.open(CHICKEN_DIR + file)
    except OSError:
        pass
    else:
        chicken_info['path'].append(CHICKEN_DIR + file)
        chicken_info['species'].append(2)
        chicken_info['class'].append(1)

RABBITS_DIR = '/home/jiaendong/rabbitsRatsChicken/' + 'train/' + 'rabbits/'
rabbits_name = os.listdir(RABBITS_DIR)
rabbits_name.sort(key=lambda x: int(x[5:-4]))
rabbits_info = {'path': [], 'species': [], 'class': []}
for file in rabbits_name:
    try:
        a = Image.open(RABBITS_DIR + file)
    except OSError:
        pass
    else:
        rabbits_info['path'].append(RABBITS_DIR + file)
        rabbits_info['species'].append(0)
        rabbits_info['class'].append(0)

RATS_DIR = '/home/jiaendong/rabbitsRatsChicken/' + 'train/' + 'rats/'
rats_name = os.listdir(RATS_DIR)
rats_name.sort(key=lambda x: int(x[5:-4]))
rats_info = {'path': [], 'species': [], 'class': []}
for file in rats_name:
    try:
        a = Image.open(RATS_DIR + file)
    except OSError:
        pass
    else:
        rats_info['path'].append(RATS_DIR + file)
        rats_info['species'].append(1)
        rats_info['class'].append(0)

total_info = {
    'path': [x for x in rabbits_info['path']] +
            [x for x in rats_info['path']] +
            [x for x in chicken_info['path']],
    'species':[x for x in rabbits_info['species']] +
              [x for x in rats_info['species']] +
              [x for x in chicken_info['species']],
    'class':[x for x in rabbits_info['class']] +
            [x for x in rats_info['class']] +
            [x for x in chicken_info['class']]
}
anno = pd.DataFrame(total_info)
anno.to_csv('train_anno.csv')
print('train_anno file is saved.')


CHICKEN_DIR = '/home/jiaendong/rabbitsRatsChicken/' + 'val/' + 'chicken/'
chicken_name = os.listdir(CHICKEN_DIR)
chicken_name.sort(key=lambda x: int(x[4:-4]))
chicken_info = {'path': [], 'species': [], 'class': []}

for file in chicken_name:
    try:
        a = Image.open(CHICKEN_DIR + file)
    except OSError:
        pass
    else:
        chicken_info['path'].append(CHICKEN_DIR + file)
        chicken_info['species'].append(2)
        chicken_info['class'].append(1)

RABBITS_DIR = '/home/jiaendong/rabbitsRatsChicken/' + 'val/' + 'rabbits/'
rabbits_name = os.listdir(RABBITS_DIR)
rabbits_name.sort(key=lambda x: int(x[5:-4]))
rabbits_info = {'path': [], 'species': [], 'class': []}
for file in rabbits_name:
    try:
        a = Image.open(RABBITS_DIR + file)
    except OSError:
        pass
    else:
        rabbits_info['path'].append(RABBITS_DIR + file)
        rabbits_info['species'].append(0)
        rabbits_info['class'].append(0)

RATS_DIR = '/home/jiaendong/rabbitsRatsChicken/' + 'val/' + 'rats/'
rats_name = os.listdir(RATS_DIR)
rats_name.sort(key=lambda x: int(x[5:-4]))
rats_info = {'path': [], 'species': [], 'class': []}
for file in rats_name:
    try:
        a = Image.open(RATS_DIR + file)
    except OSError:
        pass
    else:
        rats_info['path'].append(RATS_DIR + file)
        rats_info['species'].append(1)
        rats_info['class'].append(0)

total_info = {
    'path': [x for x in rabbits_info['path']] +
            [x for x in rats_info['path']] +
            [x for x in chicken_info['path']],
    'species':[x for x in rabbits_info['species']] +
              [x for x in rats_info['species']] +
              [x for x in chicken_info['species']],
    'class':[x for x in rabbits_info['class']] +
            [x for x in rats_info['class']] +
            [x for x in chicken_info['class']]
}
anno = pd.DataFrame(total_info)
anno.to_csv('val_anno.csv')
print('val_anno file is saved.')