import cv2
import numpy as np
from abc import ABC, abstractmethod
from autonomous_nav.config import FeatureDetectorConfig


class FeatureDetector(ABC):
    @abstractmethod
    def detect_features(self, gray_image: np.ndarray) -> np.ndarray | None:
        pass


class ShiTomasiDetector(FeatureDetector):
    def __init__(self, config: FeatureDetectorConfig):
        self.config = config

    def detect_features(self, gray_image: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
        )
