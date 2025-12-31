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
from autonomous_nav.hazard_avoidance import ClearanceBasedHazardAvoidance
from autonomous_nav.utils import pixels_to_cm
from autonomous_nav.visualizer import Visualizer
from autonomous_nav.mission_manager import MissionManager, MissionMode


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
        hazard_detector = ClearanceBasedHazardAvoidance(self.config.hazard, self.config)
        visualizer = Visualizer(self.config)

        # New: Mission Manager
        mission_manager = MissionManager(self.config)

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
            # Always use the ORIGINAL target for determining landing phase entry/exit and search zone
            remaining_dist_to_original = np.hypot(
                position.remaining_x, position.remaining_y
            )

            # Default remaining for visualization (will override in landing if locked)
            remaining_x_cm = position.remaining_x
            remaining_y_cm = position.remaining_y

            target_px = None
            outer_radius_cm = None

            if mission_manager.in_landing_phase:
                # Search zone is ALWAYS centered on the original target
                ppc = self.config.global_.pixels_per_cm
                target_dx_px = position.remaining_x * ppc
                target_dy_px = -position.remaining_y * ppc
                frame_center_x = w // 2
                frame_center_y = h // 2
                target_px = (
                    int(frame_center_x + target_dx_px),
                    int(frame_center_y + target_dy_px),
                )
                outer_radius_cm = self.config.navigation.arrival_outer_threshold_cm

                # But for visualization of arrow/text in LANDING_APPROACH, show distance to locked site if exists
                if mission_manager.landing_target_x_cm is not None:
                    remaining_x_cm = (
                        mission_manager.landing_target_x_cm - position.pos_x
                    )
                    remaining_y_cm = (
                        mission_manager.landing_target_y_cm - position.pos_y
                    )
                # Otherwise (e.g., NO_SAFE_ZONE), keep original remaining

            safe_dx_cm, safe_dy_cm, hazard_mask, safe_center_px, weighted_map = (
                hazard_detector.detect(
                    valid_new_pts.reshape(-1, 2),
                    h,
                    w,
                    target_px=target_px,  # Fixed on original target
                    outer_radius_cm=outer_radius_cm,
                )
            )

            # Update mission mode
            current_mode = mission_manager.update(
                position.pos_x,
                position.pos_y,
                position.vel_x,
                position.vel_y,
                safe_dx_cm,
                safe_dy_cm,
                safe_center_px,
            )

            # Visualization — pass the correct remaining_x/y (now reflects locked site if applicable)
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
                current_mode,
                weighted_map,
            )

            cv2.imshow("Martian Rover Navigation", annotated)

            old_gray = gray.copy()
            last_frame_time = current_time

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                position.reset()
                mission_manager.reset()
                trail_mask = np.zeros_like(frame)
                old_features = feature_detector.detect_features(gray)  # Redetect fresh
                print("Full system reset")

        camera.stop()
        cv2.destroyAllWindows()
