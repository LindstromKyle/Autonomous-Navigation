import time
import cv2
import numpy as np
from autonomous_nav.config import AppConfig
from autonomous_nav.camera import CameraModule
from autonomous_nav.preprocessor import (
    PreprocessorPipeline,
    CLAHEPreprocessor,
    GaussianBlurPreprocessor,
)
from autonomous_nav.feature_detector import ShiTomasiDetector
from autonomous_nav.optical_flow import OpticalFlowModule
from autonomous_nav.position_estimator import PositionEstimator
from autonomous_nav.imu import IMUModule
from autonomous_nav.hazard_avoidance import DensityBasedHazardAvoidance
from autonomous_nav.utils import pixels_to_cm
from autonomous_nav.visualizer import Visualizer
from autonomous_nav.commander import Commander


class AutonomousNavigationApp:

    def __init__(self, config: AppConfig):
        self.config = config

    def run(self):

        camera = CameraModule(self.config)

        # Build preprocessor chain
        preprocessors = []
        if self.config.preprocessor.clahe_enabled:
            preprocessors.append(CLAHEPreprocessor(self.config.preprocessor))
        if self.config.preprocessor.gaussian_blur_enabled:
            preprocessors.append(GaussianBlurPreprocessor(self.config.preprocessor))
        preprocessor = PreprocessorPipeline(preprocessors)

        feature_detector = ShiTomasiDetector(self.config.feature_detector)
        optical_flow = OpticalFlowModule(self.config.optical_flow)
        imu = IMUModule(self.config.imu)

        position = PositionEstimator(self.config)
        hazard_detector = DensityBasedHazardAvoidance(self.config.hazard, self.config)
        visualizer = Visualizer(self.config)
        commander = Commander()

        # Preview countdown
        camera.run_countdown_preview(4.0)

        # Initial frame and features
        old_frame = camera.capture_frame()
        last_frame_time = time.time()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
        old_gray = preprocessor.process(old_gray)
        old_features = feature_detector.detect_features(old_gray)
        trail_mask = np.zeros_like(old_frame)

        frame_count = 0

        print("\n=== Martian Rover Navigation ===")

        # Target prediction rate (Hz) — adjust based on Pi 5 performance
        target_predict_rate = self.config.imu.target_predict_rate
        min_update_threshold = self.config.global_.min_features // 2

        imu.last_time = time.time()

        while True:
            current_time = time.time()
            frame_dt = current_time - last_frame_time  # Actual time between frames

            frame = camera.capture_frame()
            gray_raw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = preprocessor.process(gray_raw)
            frame_count += 1

            valid_new_pts = np.empty((0, 1, 2))
            valid_old_pts = np.empty((0, 1, 2))

            if old_features is not None and len(old_features) > 0:
                new_features, status = optical_flow.track_features(
                    old_gray, gray, old_features
                )
                if new_features is not None:
                    valid_new_pts = new_features[status.ravel() == 1]
                    valid_old_pts = old_features[status.ravel() == 1]

            flow_dx_px, flow_dy_px = optical_flow.compute_median_flow(
                new_features,
                status if "status" in locals() else np.array([]),
                old_features,
            )

            # === HIGH-RATE PREDICTION USING MULTIPLE IMU READINGS ===
            # We aim to run predict() many times with small dt
            num_predict_steps = max(1, int(frame_dt * target_predict_rate))
            successful_predicts = 0

            for _ in range(num_predict_steps):
                imu_data = imu.read()
                if imu_data and imu_data["dt"] > 0:
                    # Predict with small time step using latest accel
                    position.predict(imu_data["accel"], imu_data["dt"])
                    successful_predicts += 1

            # Fallback: if no IMU data at all, use one big step with zero accel
            if successful_predicts == 0 and frame_dt > 0:
                position.predict(np.zeros(3), frame_dt)

            # === VISUAL VELOCITY MEASUREMENT AND UPDATE ===
            if frame_dt > 0:
                vis_vel_x = (
                    -pixels_to_cm(flow_dx_px, self.config.global_.pixels_per_cm)
                    / frame_dt
                )
                vis_vel_y = (
                    pixels_to_cm(flow_dy_px, self.config.global_.pixels_per_cm)
                    / frame_dt
                )
            else:
                vis_vel_x = vis_vel_y = 0.0

            # Only perform update if we have reliable visual data
            if len(valid_new_pts) >= min_update_threshold:
                position.update(vis_vel_x, vis_vel_y)
            else:
                # Optional: print warning less frequently
                if frame_count % 30 == 0:
                    print(
                        f"Low features ({len(valid_new_pts)}), skipping visual update — relying on IMU"
                    )

            # Update tracked features
            old_features = valid_new_pts if valid_new_pts.size > 0 else None

            # Redetect features if needed
            if (
                old_features is None
                or len(old_features) < self.config.global_.min_features
                or frame_count % self.config.global_.num_frames_redetect == 0
            ):
                old_features = feature_detector.detect_features(gray)
                trail_mask = np.zeros_like(old_frame)

            # Hazard detection (uses current valid features)
            h, w = gray.shape
            safe_dx_cm, safe_dy_cm, hazard_mask, safe_center_px = (
                hazard_detector.detect(valid_new_pts.reshape(-1, 2), h, w)
            )

            # Visualization
            annotated = visualizer.annotate_frame(
                frame.copy(),
                trail_mask,
                valid_new_pts.reshape(-1, 2),
                valid_old_pts.reshape(-1, 2),
                *position.position,
                len(valid_new_pts),
                hazard_mask,
                safe_center_px,
            )

            cv2.imshow("Martian Rover Navigation", annotated)

            # Console commands every 30 frames
            if frame_count % 30 == 0:
                # Use raw flow for commands (consistent with original behavior)
                flow_dx_cm = -(flow_dx_px / self.config.global_.pixels_per_cm)
                flow_dy_cm = flow_dy_px / self.config.global_.pixels_per_cm
                commander.issue_commands(flow_dx_cm, flow_dy_cm, safe_dx_cm, safe_dy_cm)

            old_gray = gray.copy()
            last_frame_time = current_time  # Use the time at start of this frame

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                position.reset()
                print("Position reset")

        camera.stop()
        cv2.destroyAllWindows()
