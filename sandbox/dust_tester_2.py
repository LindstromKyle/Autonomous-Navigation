import cv2
from autonomous_nav.dust import (
    DustSimulator,
)
from autonomous_nav.feature_detector import ShiTomasiDetector
from autonomous_nav.config import AppConfig
from ultralytics import YOLO

frame = cv2.imread("/home/kyle/mars_rocks/images/train/training_img_16.jpg")


config = AppConfig()
corner_detector = ShiTomasiDetector(config.feature_detector)
model = YOLO("/home/kyle/repos/Autonomous-Navigation/examples/AI/mars_rocks_custom.pt")

dust_sim = DustSimulator(config)


dust_rgb = dust_sim.apply_dust(frame)
dust_gray = cv2.cvtColor(dust_rgb, cv2.COLOR_BGR2GRAY)
dust_features = corner_detector.detect_features(dust_gray)

cv2.imshow("Clean", frame)
cv2.imshow("Noisy", dust_rgb)

dust_corners = dust_rgb.copy()
if dust_features is not None:
    for pt in dust_features.reshape(-1, 2):
        cv2.circle(dust_corners, tuple(map(int, pt)), 3, (0, 0, 255), -1)
cv2.imshow("Noisy Features", dust_corners)

dust_yolo = dust_rgb.copy()
results = model(frame, imgsz=320, conf=0.3, verbose=False)[0]
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    color = (0, 0, 255)
    cv2.rectangle(dust_yolo, (x1, y1), (x2, y2), color, 1)
cv2.imshow("Noisy Yolo", dust_yolo)

cv2.waitKey(0)
cv2.destroyAllWindows()
