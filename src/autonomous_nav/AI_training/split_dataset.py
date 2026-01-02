import os
import shutil
import random
from pathlib import Path

# ================== Configuration ==================
# Root directories
DATASET_ROOT = "/home/kyle/marsdata_v2/yolo_dataset_grayscale"  # Change if your structure is different

TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "images/train")
TRAIN_LBL_DIR = os.path.join(DATASET_ROOT, "labels/train")

VAL_IMG_DIR = os.path.join(DATASET_ROOT, "images/val")
VAL_LBL_DIR = os.path.join(DATASET_ROOT, "labels/val")

TEST_IMG_DIR = os.path.join(DATASET_ROOT, "images/test")
TEST_LBL_DIR = os.path.join(DATASET_ROOT, "labels/test")

# Split percentages
VAL_PERCENT = 10.0
TEST_PERCENT = 10.0
TRAIN_PERCENT = 100.0 - VAL_PERCENT - TEST_PERCENT

# Random seed for reproducibility
RANDOM_SEED = 42
# ==================================================


def main():
    random.seed(RANDOM_SEED)

    # Create output directories
    for dir_path in [VAL_IMG_DIR, VAL_LBL_DIR, TEST_IMG_DIR, TEST_LBL_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # Get all image files in train folder
    image_files = [f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(".png")]

    if not image_files:
        print("No images found in training directory!")
        return

    print(f"Found {len(image_files)} images. Splitting...")

    # Shuffle
    random.shuffle(image_files)

    total = len(image_files)
    num_val = int(total * VAL_PERCENT / 100.0)
    num_test = int(total * TEST_PERCENT / 100.0)
    num_train = total - num_val - num_test

    # Split lists
    val_images = image_files[:num_val]
    test_images = image_files[num_val : num_val + num_test]
    train_images = image_files[num_val + num_test :]

    def move_set(image_list, img_dest_dir, lbl_dest_dir):
        for img in image_list:
            src_img = os.path.join(TRAIN_IMG_DIR, img)
            dst_img = os.path.join(img_dest_dir, img)
            shutil.move(src_img, dst_img)

            # Move corresponding label if it exists
            label_name = Path(img).stem + ".txt"
            src_lbl = os.path.join(TRAIN_LBL_DIR, label_name)
            dst_lbl = os.path.join(lbl_dest_dir, label_name)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)

    # Move files
    print(f"Moving {num_val} images to validation set...")
    move_set(val_images, VAL_IMG_DIR, VAL_LBL_DIR)

    print(f"Moving {num_test} images to test set...")
    move_set(test_images, TEST_IMG_DIR, TEST_LBL_DIR)

    # Remaining stay in train (no move needed)
    print(f"Leaving {num_train} images in training set.")

    print("Split complete!")
    print(f"Train: {num_train}, Val: {num_val}, Test: {num_test}")


if __name__ == "__main__":
    main()
