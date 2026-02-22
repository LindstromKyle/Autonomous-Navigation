from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GlobalConfig:
    focal_length_px: float = 975.0  # Camera focal length
    frame_size: Tuple[int, int] = (640, 480)  # Algorithm input frame size
    min_features: int = 40  # Redetect features if they fall below this number
    num_frames_redetect: int = 20  # Redetect features after this many frames
    countdown_duration: float = 3.5  # How long to run countdown preview
    initial_height: float = 45.0  # Drone cruising altitude
    final_height: float = 35.0  # Drone landing hover altitude
    save_frames: bool = True  # Whether to save frames from the scenario run


@dataclass
class PreprocessorConfig:
    clahe_enabled: bool = True  # Enable CLAHE
    clahe_clip_limit_normalized: float = 0.01  # CLAHE clit limit
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)  # Size of each CLAHE tile
    gaussian_blur_enabled: bool = False  # Enable gaussian blur
    gaussian_ksize: Tuple[int, int] = (11, 11)  # Gaussian kernel size in (x,y)
    gaussian_sigma: float = 2.0  # Gaussian kernel standard deviation


@dataclass
class DustSimulatorConfig:
    map_scale: float = 5.0  # How much larger the static map is than a frame
    correlation_distance: int = 50  # Num pixels to correlate noise over
    vel_x: float = 1.0  # Pixels per frame horizontal drift
    vel_y: float = 0.5  # Pixels per frame vertical drift
    dust_intensity: float = 0.5  # Overall blend strength [0–1]
    dust_contrast: float = 2.0  # Contrast between bright and dark regions of dust
    particle_density: float = 0.001  # Num particles per pixel
    particle_size: float = 3.0  # Dust particle size


@dataclass
class FeatureDetectorConfig:
    max_corners: int = 200  # Maximum number of features to detect
    quality_level: float = 0.2  # Eigenvalue ratio for corner detection
    min_distance: int = 7  # Min distance between detected features
    block_size: int = 7  # Size of neighborhood for each pixel to calculate variance


@dataclass
class OpticalFlowConfig:
    window_size: Tuple[int, int] = (15, 15)
    max_level: int = 2  #


@dataclass
class IMUConfig:
    sample_rate_hz: int = 250  # IMU sample rate (if reading raw)
    accel_sensitivity: float = 16384.0  # Accelerometer sensitivity
    gyro_sensitivity: float = 131.0  # Gyroscope sensitivity
    mag_sensitivity: float = 0.15  # Magnetometer sensitivity
    bias_calibration_samples: int = 500  # Number of samples to collect for bias cal
    bias_file = "/home/kyle/repos/Autonomous-Navigation/examples/imu_biases.npz"


@dataclass
class LidarConfig:
    distance_mode: int = 1  # Mode to select range gates
    timing_budget: int = 50  # Timing resolution
    cache_measurements: int = 5  # Num measurements to cache


@dataclass
class NavigationConfig:
    planned_route_dx: float = 30.0  # Distance to target zone in X
    planned_route_dy: float = 15.0  # Distance to target zone in Y
    search_zone_inner_thresh: float = 2.0  # Inner radius for arrival to search zone
    search_zone_outer_thresh: float = 15.0  # Outer radius for leaving search zone
    arrow_max_length_px: int = 250  # Max length of navigation arrow in pixels

    hover_duration_s: float = 2.0  # Time to hover within tolerance for landed
    pos_tolerance_cm: float = 2.0  # Position tolerance for hover/landed
    landing_mode_stability_frames: int = (
        30  # Num consecutive frames to declare safe zone
    )


@dataclass
class HazardConfig:
    min_clearance_cm: int = 4  # Minimum distance to hazard for a spot to be safe
    hazard_dilation_px: int = 5  # Dilate features to represent hazard radius
    proximity_bias_weight: float = 0.5  # Weight for target proximity in selection (0-1)


@dataclass
class AppConfig:
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    dust_simulator: DustSimulatorConfig = field(default_factory=DustSimulatorConfig)
    feature_detector: FeatureDetectorConfig = field(
        default_factory=FeatureDetectorConfig
    )
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
