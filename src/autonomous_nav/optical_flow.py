import numpy as np
import cv2
from autonomous_nav.config import OpticalFlowConfig


class OpticalFlowModule:
    """
    Optical flow velocity estimation module
    """

    def __init__(self, config: OpticalFlowConfig):
        self.config = config

    def track_features(
        self, old_gray: np.ndarray, new_gray: np.ndarray, old_features: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """
        Track features from one frame to the next
        """
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

    def compute_median_flow(
        self,
        new_features: np.ndarray | None,
        status: np.ndarray,
        old_features: np.ndarray,
    ) -> tuple[float, float]:
        """
        Using the tracked features, compute the median flow velocity
        """
        if new_features is None or len(new_features) == 0:
            return 0.0, 0.0

        valid_new = new_features[status.ravel() == 1]
        valid_old = old_features[status.ravel() == 1]

        if len(valid_new) <= 5:
            return 0.0, 0.0

        flow = valid_new.reshape(-1, 2) - valid_old.reshape(-1, 2)
        return np.median(flow[:, 0]), np.median(flow[:, 1])
