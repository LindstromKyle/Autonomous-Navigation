import os
import cv2
from pathlib import Path

# ================== Config ==================
RGB_FOLDER = Path("/home/kyle/marsdata_v2/raw_marsdata_v2/images")
GRAY_FOLDER = Path("/home/kyle/marsdata_v2/raw_marsdata_v2/images_grayscale")

# Supported image extensions (add more if needed)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
# ===========================================


def convert_to_grayscale():
    # Create grayscale folder if it doesn't exist
    GRAY_FOLDER.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = [
        p
        for p in RGB_FOLDER.iterdir()
        if p.suffix.lower() in IMG_EXTENSIONS and p.is_file()
    ]

    if not image_files:
        print("No images found in the source folder!")
        return

    print(f"Found {len(image_files)} images. Converting to grayscale...\n")

    successful = 0
    failed = 0

    for img_path in image_files:
        try:
            # Read image in color (needed for proper handling)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read (corrupted?): {img_path.name}")
                failed += 1
                continue

            # Convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Save to grayscale folder with same filename
            gray_path = GRAY_FOLDER / img_path.name
            cv2.imwrite(str(gray_path), gray_img)

            successful += 1
            if successful % 50 == 0:
                print(f"Processed {successful}/{len(image_files)} images...")

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            failed += 1

    print("\n=== Conversion Complete ===")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"Grayscale images saved to: {GRAY_FOLDER}")


if __name__ == "__main__":
    convert_to_grayscale()
