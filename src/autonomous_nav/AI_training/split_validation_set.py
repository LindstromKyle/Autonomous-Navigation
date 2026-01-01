import os
import shutil
import random

TRAIN_IMG_DIR = "/home/kyle/marsdata_v2/yolo_dataset/images/train/"
TRAIN_LBL_DIR = "/home/kyle/marsdata_v2/yolo_dataset/labels/train/"
VAL_IMG_DIR = "/home/kyle/marsdata_v2/yolo_dataset/images/val/"
VAL_LBL_DIR = "/home/kyle/marsdata_v2/yolo_dataset/labels/val/"

os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(VAL_LBL_DIR, exist_ok=True)

all_images = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(".png")]
random.shuffle(all_images)
val_images = all_images[:50]  # ~20% for val

for img in val_images:
    shutil.move(os.path.join(TRAIN_IMG_DIR, img), os.path.join(VAL_IMG_DIR, img))
    lbl = img.rsplit(".", 1)[0] + ".txt"
    if os.path.exists(os.path.join(TRAIN_LBL_DIR, lbl)):
        shutil.move(os.path.join(TRAIN_LBL_DIR, lbl), os.path.join(VAL_LBL_DIR, lbl))
