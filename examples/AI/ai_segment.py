import cv2
import time
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from autonomous_nav.preprocessor import CLAHEPreprocessor, GaussianBlurPreprocessor
from autonomous_nav.config import PreprocessorConfig

# ================== Config ==================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INPUT_SIZE = 320

# NEW: Preprocessing options
FORCE_RGB_CONVERSION = False  # Always convert frame to RGB before inference
APPLY_CLAHE_PER_CHANNEL = False  # Apply CLAHE independently on R, G, B channels
PREPROCESSOR_CONFIG = PreprocessorConfig()
# ===========================================


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply selected preprocessing steps to the input frame.
    """
    processed = frame.copy()

    # Option 1: Ensure RGB (Picamera2 already gives RGB888, but this makes it robust)
    if FORCE_RGB_CONVERSION and processed.shape[2] == 3:
        # If somehow in BGR (e.g., from other sources), convert to RGB
        if cv2.cvtColor(processed, cv2.COLOR_BGR2RGB).sum() != processed.sum():
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    # Option 2: Apply CLAHE on each channel separately
    if APPLY_CLAHE_PER_CHANNEL:
        clahe = CLAHEPreprocessor(PREPROCESSOR_CONFIG)
        blur = GaussianBlurPreprocessor(PREPROCESSOR_CONFIG)
        # Split into channels, apply CLAHE, merge back
        channels = cv2.split(processed)
        clahe_channels = [clahe.process(ch) for ch in channels]
        blurred_channels = [blur.process(ch) for ch in clahe_channels]
        processed = cv2.merge(blurred_channels)

    return processed


def main():
    preprocessor = CLAHEPreprocessor(PREPROCESSOR_CONFIG)
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    print("Camera started. Loading YOLOv8 detection model...")

    model = YOLO(
        "/home/kyle/repos/Autonomous-Navigation/examples/AI/full_marsdata_v2.pt"
    )
    print("Model loaded. Starting live detection (press 'q' to quit)")

    # Warm up model
    dummy = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    # dummy = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)

    model(dummy, imgsz=INPUT_SIZE, verbose=False)

    prev_time = time.time()
    while True:
        frame = picam2.capture_array()  # Shape: (H, W, 3), RGB888
        # gray_raw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # gray = preprocessor.process(gray_raw)
        # === Preprocess frame according to config ===
        # input_frame = preprocess_frame(frame)

        # Run inference
        results = model(frame, imgsz=INPUT_SIZE, conf=0.005, verbose=False)[0]

        # Draw bounding boxes on original frame (for clean visualization)
        overlay = frame.copy()
        hazard_centers = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = f"{results.names[cls_id]} {conf:.2f}"

            color = (0, 0, 255)  # Red for bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # Centroid
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            hazard_centers.append((center_x, center_y))
            cv2.circle(overlay, (center_x, center_y), 6, (255, 0, 0), -1)  # Blue center

        # FPS and detection count
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        info_text = f"FPS: {fps:.1f} | Detections: {len(hazard_centers)}"
        if APPLY_CLAHE_PER_CHANNEL:
            info_text += " | CLAHE(per-ch)"
        if FORCE_RGB_CONVERSION:
            info_text += " | RGB"

        cv2.putText(
            overlay,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.imshow("YOLOv8 Bounding Box Detection Demo", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
