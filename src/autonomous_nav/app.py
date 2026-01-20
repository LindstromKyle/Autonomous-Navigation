import time
import cv2
import numpy as np
from autonomous_nav.config import AppConfig
from autonomous_nav.camera import CameraModule
from autonomous_nav.lidar import LidarModule
from autonomous_nav.preprocessor import (
    PreprocessorPipeline,
    CLAHEPreprocessor,
    GaussianBlurPreprocessor,
)
from autonomous_nav.feature_detector import ShiTomasiDetector
from autonomous_nav.optical_flow import OpticalFlowModule
from autonomous_nav.state_estimator import StateEstimator
from autonomous_nav.imu import IMUModule
from autonomous_nav.hazard_avoidance import ClearanceBasedHazardAvoidance
from autonomous_nav.utils import cm_to_pixels, pixels_to_cm
from autonomous_nav.visualizer import Visualizer
from autonomous_nav.mission_manager import MissionManager


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
        state = StateEstimator(self.config)

        lidar = LidarModule(self.config.lidar)
        hazard_detector = ClearanceBasedHazardAvoidance(self.config.hazard, self.config)
        visualizer = Visualizer(self.config)

        # New: Mission Manager
        mission_manager = MissionManager(self.config)

        # Preview countdown
        camera.run_countdown_preview()

        # imu = IMUModule(self.config.imu)
        # state.state[6:10] = imu.init_quat.copy()
        # state.normalize_quaternion()

        # Initial frame and features
        old_frame = camera.capture_frame()
        last_frame_time = time.time()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
        old_gray = preprocessor.process(old_gray)
        old_features = feature_detector.detect_features(old_gray)
        trail_mask = np.zeros_like(old_frame)

        frame_count = 0

        print("\n=== Martian Rover Navigation ===")

        # imu.last_time = time.time()
        time.sleep(0.02)
        while True:
            current_time = time.time()
            frame_dt = max(current_time - last_frame_time, 0.025)

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

            # IMU predict state
            # imu_data = imu.read()
            # if imu_data:
            #     state.predict(imu_data["accel"], imu_data["gyro"], imu_data["dt"])
            state.predict(frame_dt)

            # Visual velocity update
            vis_vel_x = (
                -pixels_to_cm(
                    flow_dx_px, state.position[2], self.config.global_.focal_length_px
                )
                / frame_dt
            )
            vis_vel_y = (
                pixels_to_cm(
                    flow_dy_px, state.position[2], self.config.global_.focal_length_px
                )
                / frame_dt
            )
            flow_mag = np.hypot(vis_vel_x, vis_vel_y)
            if flow_mag < 0.5 and len(valid_new_pts) > 20:
                # Zero-Velocity Update (ZUPT)
                state.update_visual(0.0, 0.0)  # Forces velocity toward zero
            else:
                state.update_visual(vis_vel_x, vis_vel_y)

            # Lidar range update
            lidar_data = lidar.read()
            if lidar_data:
                state.update_lidar(lidar_data["distance_cm"])

            old_features = valid_new_pts if valid_new_pts.size > 0 else None

            if (
                old_features is None
                or len(old_features) < self.config.global_.min_features
                or frame_count % self.config.global_.num_frames_redetect == 0
            ):
                old_features = feature_detector.detect_features(gray)
                trail_mask = np.zeros_like(old_frame)

            dx_to_search_center = state.dx_to_search_center
            dy_to_search_center = state.dy_to_search_center

            if mission_manager.in_landing_phase:
                # Search zone is ALWAYS centered on the original target
                dx_to_search_center = cm_to_pixels(
                    dx_to_search_center,
                    state.position[2],
                    self.config.global_.focal_length_px,
                )
                dy_to_search_center = -cm_to_pixels(
                    dy_to_search_center,
                    state.position[2],
                    self.config.global_.focal_length_px,
                )
                frame_center_x = w // 2
                frame_center_y = h // 2
                search_zone_center = (
                    int(frame_center_x + dx_to_search_center),
                    int(frame_center_y + dy_to_search_center),
                )
                search_zone_outer_thresh = (
                    self.config.navigation.search_zone_outer_thresh
                )

                (
                    current_frame_safe_dx,
                    current_frame_safe_dy,
                    current_frame_safe_px,
                    weighted_map,
                ) = hazard_detector.compute_safe_zone(
                    valid_new_pts.reshape(-1, 2),
                    h,
                    w,
                    state.position[2],
                    search_zone_center=search_zone_center,
                    search_zone_outer_thresh=search_zone_outer_thresh,
                    frame=frame,
                )
                hazard_points = hazard_detector.hazard_points

            else:
                current_frame_safe_dx = current_frame_safe_dy = None
                current_frame_safe_px = None
                weighted_map = None
                hazard_points = None

            # Update mission mode
            mission_manager.update(
                state.position[0],
                state.position[1],
                state.position[2],
                current_frame_safe_dx,
                current_frame_safe_dy,
                current_frame_safe_px,
            )
            if mission_manager.locked_landing_target_x_cm is not None:
                dx_to_locked_landing_target_cm = (
                    mission_manager.locked_landing_target_x_cm - state.position[0]
                )
                dy_to_locked_landing_target_cm = (
                    mission_manager.locked_landing_target_y_cm - state.position[1]
                )
            else:
                dx_to_locked_landing_target_cm = None
                dy_to_locked_landing_target_cm = None

            # Visualization
            annotated = visualizer.annotate_frame(
                frame.copy(),
                trail_mask,
                valid_new_pts.reshape(-1, 2),
                valid_old_pts.reshape(-1, 2),
                state,
                len(valid_new_pts),
                dx_to_search_center,
                dy_to_search_center,
                dx_to_locked_landing_target_cm,
                dy_to_locked_landing_target_cm,
                mission_manager,
                weighted_map,
                hazard_points,
            )

            cv2.imshow("Martian Rover Navigation", annotated)

            old_gray = gray.copy()
            last_frame_time = current_time

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                state.reset()
                mission_manager.reset()
                trail_mask = np.zeros_like(frame)
                old_features = feature_detector.detect_features(gray)  # Redetect fresh
                print("Full system reset")

        camera.stop()
        lidar.stop()
        cv2.destroyAllWindows()
