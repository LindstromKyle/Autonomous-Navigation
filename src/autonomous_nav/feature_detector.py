import cv2
import numpy as np
from abc import ABC, abstractmethod
from autonomous_nav.config import FeatureDetectorConfig


class FeatureDetector(ABC):
    """
    ABC for feature detectors
    """

    @abstractmethod
    def detect_features(self, gray_image: np.ndarray) -> np.ndarray | None:
        """
        Subclasses must implement this method
        """
        pass


class ShiTomasiDetector(FeatureDetector):
    """
    Shit-Tomasi feature detector module
    """

    def __init__(self, config: FeatureDetectorConfig):
        self.config = config

    def detect_features(self, gray_image: np.ndarray) -> np.ndarray | None:
        """
        Detects corners on an input image
        """
        return cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
        )
