import cv2
import os
import numpy as np

# ================== Config ==================
# Adjust these paths to match where you generated the YOLO dataset
IMAGE_DIR = "/home/kyle/marsdata_v2/yolo_dataset/images/train/"  # Folder with the saved RGB images
LABEL_DIR = "/home/kyle/marsdata_v2/yolo_dataset/labels/train/"  # Folder with the .txt label files

# Colors (BGR for OpenCV)
BOX_COLOR = (0, 255, 0)  # Green boxes
CENTER_COLOR = (0, 0, 255)  # Red center dot
TEXT_COLOR = (255, 255, 255)  # White text

# Display settings
WINDOW_NAME = "YOLO Dataset Bounding Box Checker"
DELAY_MS = 0  # 0 = wait for key press, >0 = auto-advance after N ms

# ===========================================


def draw_yolo_boxes(image, label_path, img_width, img_height):
    """Draw all bounding boxes from a YOLO .txt label file onto the image."""
    if not os.path.exists(label_path):
        return image  # No labels for this image

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        class_id = int(parts[0])
        cx = float(parts[1]) * img_width
        cy = float(parts[2]) * img_height
        w = float(parts[3]) * img_width
        h = float(parts[4]) * img_height

        # Convert center + w/h to corner coordinates
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Draw center point
        cv2.circle(image, (int(cx), int(cy)), 5, CENTER_COLOR, -1)

        # Optional label (class id)
        label = f"rock {class_id}"
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2
        )

    return image


def main():
    if not os.path.exists(IMAGE_DIR):
        print(f"Image directory not found: {IMAGE_DIR}")
        return
    if not os.path.exists(LABEL_DIR):
        print(f"Label directory not found: {LABEL_DIR}")
        return

    image_files = sorted(
        [
            f
            for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    print(f"Found {len(image_files)} images. Starting viewer...")
    print(
        "Controls: 'q' = quit, 'n' = next, 'p' = previous, any other key = pause until next key"
    )

    idx = 0
    while True:
        if idx < 0:
            idx = 0
        if idx >= len(image_files):
            idx = len(image_files) - 1

        img_path = os.path.join(IMAGE_DIR, image_files[idx])
        label_path = os.path.join(
            LABEL_DIR, os.path.splitext(image_files[idx])[0] + ".txt"
        )

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not load image: {img_path}")
            idx += 1
            continue

        h, w = img.shape[:2]
        annotated = draw_yolo_boxes(img.copy(), label_path, w, h)

        # Add info text
        info = f"{idx + 1}/{len(image_files)} - {image_files[idx]}"
        cv2.putText(
            annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        cv2.imshow(WINDOW_NAME, annotated)

        key = cv2.waitKey(DELAY_MS) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            idx += 1
        elif key == ord("p"):
            idx -= 1
        # Any other key just continues to next (or pauses if DELAY_MS=0)

    cv2.destroyAllWindows()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
