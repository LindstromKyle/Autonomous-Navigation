import cv2
from autonomous_nav.dust import (
    DustSimulator,
)
from autonomous_nav.feature_detector import ShiTomasiDetector
from autonomous_nav.config import AppConfig
from ultralytics import YOLO

# Load config and detectors
config = AppConfig()
corner_detector = ShiTomasiDetector(config.feature_detector)
model = YOLO("/home/kyle/repos/Autonomous-Navigation/examples/AI/mars_rocks_custom.pt")

dust_sim = DustSimulator(config)

# Load a test image (replace with your capture_frame() or file path)
frame = cv2.imread("/home/kyle/mars_rocks/images/train/training_img_09.jpg")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
clean_features = corner_detector.detect_features(gray)
print(
    f"Clean features detected: {len(clean_features) if clean_features is not None else 0}"
)

# Add noise and re-detect
dust_rgb = dust_sim.apply_dust(frame)
dust_gray = cv2.cvtColor(dust_rgb, cv2.COLOR_BGR2GRAY)
dust_features = corner_detector.detect_features(dust_gray)
print(
    f"Noisy features detected: {len(dust_features) if dust_features is not None else 0}"
)

# Visualize (optional)
cv2.imshow("Clean", frame)
cv2.imshow("Noisy", dust_rgb)
if clean_features is not None:
    for pt in clean_features.reshape(-1, 2):
        cv2.circle(frame, tuple(map(int, pt)), 4, (0, 255, 0), -1)
if dust_features is not None:
    for pt in dust_features.reshape(-1, 2):
        cv2.circle(dust_rgb, tuple(map(int, pt)), 3, (0, 0, 255), -1)
# cv2.imshow("Features", frame)
cv2.imshow("Noisy Features", dust_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
