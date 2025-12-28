import cv2
import numpy as np
from abc import ABC, abstractmethod
from config import FeatureDetectorConfig


class FeatureDetector(ABC):
    @abstractmethod
    def detect_features(self, gray_image: np.ndarray) -> np.ndarray | None:
        pass

    @abstractmethod
    def track_features(
        self, old_gray: np.ndarray, new_gray: np.ndarray, old_features: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
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

    def track_features(
        self, old_gray: np.ndarray, new_gray: np.ndarray, old_features: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            old_gray,
            new_gray,
            old_features,
            None,
            winSize=self.config.window_size,
            maxLevel=self.config.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        return new_features, status
