import time
import cv2
import numpy as np
import os
from datetime import datetime
import h5py
from autonomous_nav.config import AppConfig
from autonomous_nav.camera import CameraModule
from autonomous_nav.dust import DustSimulator
from autonomous_nav.lidar import LidarModule
from autonomous_nav.preprocessor import (
    PreprocessorPipeline,
    CLAHEPreprocessor,
    GaussianBlurPreprocessor,
)
from autonomous_nav.feature_detector import ShiTomasiDetector
from autonomous_nav.optical_flow import OpticalFlowModule
from autonomous_nav.state_estimator import InertialStateEstimator
from autonomous_nav.imu import IMUModule
from autonomous_nav.hazard_avoidance import (
    AIHazardAvoidance,
    ClearanceBasedHazardAvoidance,
)
from autonomous_nav.utils import cm_to_pixels, pixels_to_cm
from autonomous_nav.visualizer import Visualizer
from autonomous_nav.mission_manager import MissionManager


class AutonomousNavigationApp:
    """
    Main application module
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.recorded_frames = []

    def run(self):
        """
        Runs the main application loop
        """

        # Camera module
        camera = CameraModule(self.config)

        # Dust simulator module
        dust_sim = DustSimulator(self.config)

        # Preprocessing chain module
        preprocessors = []
        if self.config.preprocessor.clahe_enabled:
            preprocessors.append(CLAHEPreprocessor(self.config.preprocessor))
        if self.config.preprocessor.gaussian_blur_enabled:
            preprocessors.append(GaussianBlurPreprocessor(self.config.preprocessor))
        preprocessor = PreprocessorPipeline(preprocessors)

        # Feature detection module
        feature_detector = ShiTomasiDetector(self.config.feature_detector)

        # Optical flow module
        optical_flow = OpticalFlowModule(self.config.optical_flow)

        # State estimator module
        state = InertialStateEstimator(self.config)

        # Lidar module
        lidar = LidarModule(self.config.lidar)

        # Hazard detection module
        hazard_detector = AIHazardAvoidance(self.config.hazard, self.config)

        # Visualizer module
        visualizer = Visualizer(self.config)

        # Mission manager module
        mission_manager = MissionManager(self.config)

        # Preview countdown
        camera.run_countdown_preview()

        # IMU module
        imu = IMUModule(self.config.imu)

        # Initialize state
        state.state[6:10] = imu.init_quat.copy()
        state.normalize_quaternion()

        # Initial frame
        old_frame = camera.capture_frame()
        old_frame = dust_sim.apply_dust(old_frame)
        last_frame_time = time.time()

        # Grayscale
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)

        # Preprocess initial frame
        old_gray = preprocessor.process(old_gray)

        # Detect initial features
        old_features = feature_detector.detect_features(old_gray)
        trail_mask = np.zeros_like(old_frame)
        frame_count = 0

        print(f"\nStarting Martian Drone Navigation\n")

        # Begin loop
        imu.last_time = time.time()
        time.sleep(0.02)
        while True:

            # Calculate dt
            current_time = time.time()
            frame_dt = max(current_time - last_frame_time, 0.025)

            # Capture image
            frame = camera.capture_frame()

            # Apply dust
            frame = dust_sim.apply_dust(frame)

            # Grayscale
            gray_raw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Preprocessor chain
            gray = preprocessor.process(gray_raw)
            frame_count += 1
            h, w = gray.shape

            # Track features
            valid_new_pts = np.empty((0, 1, 2))
            valid_old_pts = np.empty((0, 1, 2))
            if old_features is not None and len(old_features) > 0:
                new_features, status = optical_flow.track_features(
                    old_gray, gray, old_features
                )
                if new_features is not None:
                    valid_new_pts = new_features[status.ravel() == 1]
                    valid_old_pts = old_features[status.ravel() == 1]

            # Compute median flow
            flow_dx_px, flow_dy_px = optical_flow.compute_median_flow(
                new_features,
                status if "status" in locals() else np.array([]),
                old_features,
            )

            # Predict state with IMU data
            imu_data = imu.read()
            if imu_data:
                state.predict(imu_data["accel"], imu_data["gyro"], imu_data["dt"])

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

            # Zero velocity update
            if flow_mag < 0.5 and len(valid_new_pts) > 20:
                state.update_visual(0.0, 0.0)
            else:
                state.update_visual(vis_vel_x, vis_vel_y)

            # Lidar range update
            lidar_data = lidar.read()
            if lidar_data:
                state.update_lidar(lidar_data["distance_cm"])

            # Redetect if needed
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

            # If in landing phase, prepare for safe zone calculation
            if mission_manager.in_landing_phase:
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

                # Compute safe zone from hazard detector
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

            # Otherwise no safe zone needed yet
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

            # Show the image
            cv2.imshow("Martian Rover Navigation", annotated)

            # Update params for next loop
            old_gray = gray.copy()
            last_frame_time = current_time

            # Add frame to saved frames list
            if self.config.global_.save_frames:
                self.recorded_frames.append((last_frame_time, annotated.copy()))

            # Quit key
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break

            # Reset key
            elif key == ord("r"):
                state.reset()
                mission_manager.reset()
                trail_mask = np.zeros_like(frame)
                old_features = feature_detector.detect_features(gray)  # Redetect fresh
                print("Full system reset")

        # Save frames
        if self.config.global_.save_frames:
            self.save_frames_to_h5()

        # Shutdown modules when finished
        camera.stop()
        lidar.stop()
        cv2.destroyAllWindows()

    def save_frames_to_h5(self):
        """
        Saves the frames from the current scenario run to HDF5
        """

        # File path
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = "/home/kyle/recordings"
        os.makedirs(out_dir, exist_ok=True)
        filename = os.path.join(out_dir, f"recording_{timestamp_str}.h5")

        # Open file
        with h5py.File(filename, "w") as f:
            n_frames = len(self.recorded_frames)
            height, width, channels = self.recorded_frames[0][1].shape

            # Frames dataset
            frames_dataset = f.create_dataset(
                "frames",
                shape=(n_frames, height, width, channels),
                dtype=np.uint8,
                chunks=(1, height // 2, width // 2, channels),
                compression="gzip",
                compression_opts=4,
            )

            # Timestamps dataset
            timestamp_dataset = f.create_dataset(
                "timestamps", shape=(n_frames,), dtype=np.float64
            )
            for i, (timestamp, frame) in enumerate(self.recorded_frames):
                frames_dataset[i] = frame
                timestamp_dataset[i] = timestamp - self.recorded_frames[0][0]

        print(f"Saved to: {filename}")
