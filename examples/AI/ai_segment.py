import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

# ================== Config ==================
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INPUT_SIZE = 320
# ===========================================


def main():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    print("Camera started. Loading YOLOv8n detection model...")

    model = YOLO("/home/kyle/repos/Autonomous-Navigation/examples/AI/best.pt")
    print("Model loaded. Starting live detection (press 'q' to quit)")

    prev_time = time.time()
    while True:
        frame = picam2.capture_array()

        # Run inference
        results = model(frame, imgsz=INPUT_SIZE, conf=0.02, verbose=False)[0]

        # Draw bounding boxes
        overlay = frame.copy()
        hazard_centers = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box corners
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            label = f"{results.names[cls_id]} {conf:.2f}"

            # Highlight as hazard (red box for demo)
            color = (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # Compute centroid for your hazard pipeline
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            hazard_centers.append((center_x, center_y))
            cv2.circle(overlay, (center_x, center_y), 5, (255, 0, 0), -1)

        # FPS display
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(
            overlay,
            f"FPS: {fps:.1f} | Detections: {len(hazard_centers)}",
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
