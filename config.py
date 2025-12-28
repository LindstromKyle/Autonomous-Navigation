from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class GlobalConfig:
    pixels_per_cm: float = 14.0
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
    max_corners: int = 200
    quality_level: float = 0.1
    min_distance: int = 7
    block_size: int = 7

    # Optical flow parameters (tightly coupled with Shi-Tomasi)
    window_size: Tuple[int, int] = (15, 15)
    max_level: int = 2


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
    hazard: HazardConfig = field(default_factory=HazardConfig)
