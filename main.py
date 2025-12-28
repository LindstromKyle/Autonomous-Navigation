import cv2
import numpy as np
from config import AppConfig
from camera import CameraModule
from preprocessor import (
    PreprocessorPipeline,
    CLAHEPreprocessor,
    GaussianBlurPreprocessor,
)
from feature_detector import ShiTomasiDetector
from optical_flow import OpticalFlowModule
from position_estimator import PositionEstimator
from hazard_avoidance import DensityBasedHazardAvoidance
from visualizer import Visualizer
from commander import Commander


def main():
    config = AppConfig()  # Or load from YAML

    camera = CameraModule(config)

    # Build preprocessor chain
    preprocessors = []
    if config.preprocessor.clahe_enabled:
        preprocessors.append(CLAHEPreprocessor(config.preprocessor))
    if config.preprocessor.gaussian_blur_enabled:
        preprocessors.append(GaussianBlurPreprocessor(config.preprocessor))
    preprocessor = PreprocessorPipeline(preprocessors)

    feature_detector = ShiTomasiDetector(config.feature_detector)
    optical_flow = OpticalFlowModule(config)
    position = PositionEstimator(config)
    hazard_detector = DensityBasedHazardAvoidance(config.hazard, config)
    visualizer = Visualizer(config)
    commander = Commander()

    # Initial frame and features
    old_frame = camera.capture_frame()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
    old_gray = preprocessor.process(old_gray)
    old_features = feature_detector.detect_features(old_gray)
    trail_mask = np.zeros_like(old_frame)

    frame_count = 0
    print("\n=== Martian Rover Navigation ===")
    print("Hybrid config + modular design active.\n")

    while True:
        frame = camera.capture_frame()
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = preprocessor.process(gray_raw)
        frame_count += 1

        valid_new_pts = np.empty((0, 1, 2))
        valid_old_pts = np.empty((0, 1, 2))

        if old_features is not None and len(old_features) > 0:
            new_features, status = feature_detector.track_features(
                old_gray, gray, old_features
            )
            if new_features is not None:
                valid_new_pts = new_features[status.ravel() == 1]
                valid_old_pts = old_features[status.ravel() == 1]

        flow_dx_px, flow_dy_px = optical_flow.compute_median_flow(
            new_features, status if "status" in locals() else np.array([]), old_features
        )
        position.update(flow_dx_px, flow_dy_px)

        # Update to tracked features first (default to continuing with valid tracked points)
        old_features = valid_new_pts if valid_new_pts.size > 0 else None

        # Refresh features if needed (check after updating to tracked state)
        if (
            old_features is None
            or len(old_features) < config.global_.min_features
            or frame_count % config.global_.num_frames_redetect == 0
        ):
            old_features = feature_detector.detect_features(gray)
            trail_mask = np.zeros_like(old_frame)

        # Hazard detection
        h, w = gray.shape
        safe_dx_cm, safe_dy_cm, hazard_mask, safe_center_px = hazard_detector.detect(
            valid_new_pts.reshape(-1, 2), h, w
        )

        # Visualisation
        annotated = visualizer.annotate_frame(
            frame.copy(),
            trail_mask,
            valid_new_pts.reshape(-1, 2),
            valid_old_pts.reshape(-1, 2),
            *position.position,
            len(valid_new_pts),
            hazard_mask,
            safe_center_px
        )

        cv2.imshow("Martian Rover Navigation", annotated)

        # Console commands every 30 frames
        if frame_count % 30 == 0:
            flow_dx_cm = -(flow_dx_px / config.global_.pixels_per_cm)
            flow_dy_cm = flow_dy_px / config.global_.pixels_per_cm
            commander.issue_commands(flow_dx_cm, flow_dy_cm, safe_dx_cm, safe_dy_cm)

        old_gray = gray.copy()

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            position.reset()
            print("Position reset")

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
