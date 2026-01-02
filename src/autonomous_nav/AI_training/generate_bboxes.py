import cv2
import numpy as np
import os
from tqdm import tqdm

# Directories
IMAGE_DIR = "/home/kyle/marsdata_v2/raw_marsdata_v2/images"
MASK_DIR = "/home/kyle/marsdata_v2/raw_marsdata_v2/masks"
OUTPUT_IMG_DIR = "/home/kyle/marsdata_v2/yolo_dataset/images/train/"
OUTPUT_LBL_DIR = "/home/kyle/marsdata_v2/yolo_dataset/labels/train/"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png"))])

for img_file in tqdm(image_files):
    img_path = os.path.join(IMAGE_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, img_file)

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"Skipping {img_file}")
        continue

    h, w = mask.shape
    # Threshold the mask
    _, binary = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, bw, bh = cv2.boundingRect(cnt)
        center_x = (x + bw / 2) / w
        center_y = (y + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h
        label_lines.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    # Save image and label
    cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, img_file), img)
    with open(
        os.path.join(OUTPUT_LBL_DIR, img_file.rsplit(".", 1)[0] + ".txt"), "w"
    ) as f:
        f.write("\n".join(label_lines))
