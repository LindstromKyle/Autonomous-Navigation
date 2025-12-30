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
        camera.run_countdown_preview()

        # Initial frame and features
        old_frame = camera.capture_frame()
        last_frame_time = time.time()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
        old_gray = preprocessor.process(old_gray)
        old_features = feature_detector.detect_features(old_gray)
        trail_mask = np.zeros_like(old_frame)

        frame_count = 0

        print("\n=== Martian Rover Navigation ===")

        target_predict_rate = self.config.imu.target_predict_rate
        min_update_threshold = self.config.global_.min_features // 2

        imu.last_time = time.time()

        while True:
            current_time = time.time()
            frame_dt = current_time - last_frame_time

            frame = camera.capture_frame()
            gray_raw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = preprocessor.process(gray_raw)
            frame_count += 1

            h, w = gray.shape

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

            # High-rate IMU prediction
            num_predict_steps = max(1, int(frame_dt * target_predict_rate))
            successful_predicts = 0
            for _ in range(num_predict_steps):
                imu_data = imu.read()
                if imu_data and imu_data["dt"] > 0:
                    position.predict(imu_data["accel"], imu_data["dt"])
                    successful_predicts += 1
            if successful_predicts == 0 and frame_dt > 0:
                position.predict(np.zeros(3), frame_dt)

            # Visual velocity update
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

            if len(valid_new_pts) >= min_update_threshold:
                position.update(vis_vel_x, vis_vel_y)
            else:
                if frame_count % 30 == 0:
                    print(
                        f"Low features ({len(valid_new_pts)}), skipping visual update — relying on IMU"
                    )

            old_features = valid_new_pts if valid_new_pts.size > 0 else None

            if (
                old_features is None
                or len(old_features) < self.config.global_.min_features
                or frame_count % self.config.global_.num_frames_redetect == 0
            ):
                old_features = feature_detector.detect_features(gray)
                trail_mask = np.zeros_like(old_frame)

            # --- Planned route logic ---
            remaining_x_cm = position.remaining_x
            remaining_y_cm = position.remaining_y
            in_landing_mode = position.in_landing_mode

            target_px = None
            if in_landing_mode:
                ppc = self.config.global_.pixels_per_cm
                target_dx_px = remaining_x_cm * ppc
                # Flip Y to align with physical coordinate system
                target_dy_px = -remaining_y_cm * ppc
                frame_center_x = w // 2
                frame_center_y = h // 2
                target_px = (
                    int(frame_center_x + target_dx_px),
                    int(frame_center_y + target_dy_px),
                )

            # Hazard detection with optional target bias
            safe_dx_cm, safe_dy_cm, hazard_mask, safe_center_px = (
                hazard_detector.detect(
                    valid_new_pts.reshape(-1, 2), h, w, target_px=target_px
                )
            )

            # Visualization (pass remaining for arrow)
            annotated = visualizer.annotate_frame(
                frame.copy(),
                trail_mask,
                valid_new_pts.reshape(-1, 2),
                valid_old_pts.reshape(-1, 2),
                *position.position,
                len(valid_new_pts),
                hazard_mask,
                safe_center_px,
                remaining_x_cm,
                remaining_y_cm,
                in_landing_mode,
            )

            cv2.imshow("Martian Rover Navigation", annotated)

            # Console commands every 30 frames
            if frame_count % 30 == 0:
                flow_dx_cm = -(flow_dx_px / self.config.global_.pixels_per_cm)
                flow_dy_cm = flow_dy_px / self.config.global_.pixels_per_cm
                Commander.issue_commands(
                    flow_dx_cm,
                    flow_dy_cm,
                    safe_dx_cm,
                    safe_dy_cm,
                    remaining_x_cm,
                    remaining_y_cm,
                    in_landing_mode,
                )

            old_gray = gray.copy()
            last_frame_time = current_time

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                position.reset()
                print("Position reset")

        camera.stop()
        cv2.destroyAllWindows()
