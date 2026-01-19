from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GlobalConfig:
    pixels_per_cm: float = 25.0  # 55cm = 17.5, 40cm = 25.0
    frame_size: Tuple[int, int] = (640, 480)
    min_features: int = 40
    num_frames_redetect: int = 20
    countdown_duration = 3.5


@dataclass
class PreprocessorConfig:
    clahe_enabled: bool = True
    clahe_clip_limit_normalized: float = 0.01
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    gaussian_blur_enabled: bool = False
    gaussian_ksize: Tuple[int, int] = (11, 11)  # (3, 3)
    gaussian_sigma: float = 2.0


@dataclass
class FeatureDetectorConfig:
    max_corners: int = 200
    quality_level: float = 0.2
    min_distance: int = 7
    block_size: int = 7


@dataclass
class OpticalFlowConfig:
    window_size: Tuple[int, int] = (15, 15)
    max_level: int = 2


@dataclass
class IMUConfig:
    sample_rate_hz: int = 250
    accel_sensitivity: float = 16384.0
    gyro_sensitivity: float = 131.0
    mag_sensitivity: float = 0.15
    bias_calibration_samples: int = 500
    bias_file = "/home/kyle/repos/Autonomous-Navigation/examples/imu_biases.npz"


@dataclass
class LidarConfig:
    distance_mode: int = 1
    timing_budget: int = 50
    cache_measurements: int = 5


@dataclass
class NavigationConfig:
    planned_route_dx: float = 10.0
    planned_route_dy: float = 0.0
    search_zone_inner_thresh: float = 2.0
    search_zone_outer_thresh: float = 10.0
    arrow_max_length_px: int = 250

    hover_duration_s: float = 2.0  # Time to hover within tolerance for landed
    pos_tolerance_cm: float = 2.0  # Position tolerance for hover/landed
    vel_tolerance_cm_s: float = 2.0  # Velocity tolerance for hover
    landing_mode_stability_frames: int = 8


@dataclass
class HazardConfig:
    grid_size: int = 8
    threshold: int = 2
    exclude_boundaries: bool = True

    min_clearance_cm: int = 4  # Minimum distance to hazard for a spot to be safe
    hazard_dilation_px: int = 5  # Dilate features to represent hazard radius
    proximity_bias_weight: float = 0.5  # Weight for target proximity in selection (0-1)


@dataclass
class AppConfig:
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    feature_detector: FeatureDetectorConfig = field(
        default_factory=FeatureDetectorConfig
    )
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
