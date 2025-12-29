from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GlobalConfig:
    pixels_per_cm: float = 25.0  # 55cm = 17.5, 40cm = 25.0
    frame_size: Tuple[int, int] = (640, 480)
    min_features: int = 40
    num_frames_redetect: int = 20


@dataclass
class PreprocessorConfig:
    clahe_enabled: bool = True
    clahe_clip_limit_normalized: float = 0.01
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    gaussian_blur_enabled: bool = False
    gaussian_ksize: Tuple[int, int] = (3, 3)
    gaussian_sigma: float = 0.0


@dataclass
class FeatureDetectorConfig:
    max_corners: int = 300
    quality_level: float = 0.1
    min_distance: int = 7
    block_size: int = 7


@dataclass
class OpticalFlowConfig:
    window_size: Tuple[int, int] = (15, 15)
    max_level: int = 2


@dataclass
class IMUConfig:
    sample_rate_hz: int = 20  # Match camera FPS-ish
    accel_sensitivity: float = 16384.0
    gyro_sensitivity: float = 131.0
    mag_sensitivity: float = 0.15
    bias_calibration_samples: int = 100
    target_predict_rate: int = 100
    bias_file = "/home/kyle/repos/Autonomous-Navigation/examples/imu_biases.npz"


@dataclass
class HazardConfig:
    grid_size: int = 8
    threshold: int = 2
    exclude_boundaries: bool = True


@dataclass
class AppConfig:
    global_: GlobalConfig = field(default_factory=GlobalConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    feature_detector: FeatureDetectorConfig = field(
        default_factory=FeatureDetectorConfig
    )
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
