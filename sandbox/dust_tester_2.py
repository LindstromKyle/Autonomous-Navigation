from autonomous_nav.camera import CameraModule
from autonomous_nav.config import AppConfig
from autonomous_nav.dust import DustSimulator
from autonomous_nav.feature_detector import ShiTomasiDetector
from autonomous_nav.preprocessor import CLAHEPreprocessor, PreprocessorPipeline


config = AppConfig()

camera = CameraModule(config)

dust_sim = DustSimulator(
    frame_shape=config.global_.frame_size[::-1],
    map_scale=3.0,
    correlation_distance=50,
    vel_x=2.0,
    vel_y=1.0,
    dust_intensity=0.8,
    gaussian_std=12.0,
    gaussian_intensity=1.0,
)

# Build preprocessor chain
preprocessors = []
preprocessors.append(CLAHEPreprocessor(config.preprocessor))
preprocessor = PreprocessorPipeline(preprocessors)

feature_detector = ShiTomasiDetector(config.feature_detector)

while True:

    frame = camera.capture_frame()
    frame = dust_sim.apply_dust(frame)
