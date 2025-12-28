import numpy as np
from config import AppConfig


class OpticalFlowModule:
    def __init__(self, config: AppConfig):
        self.pixels_per_cm = config.global_.pixels_per_cm

    def compute_median_flow(
        self,
        new_features: np.ndarray | None,
        status: np.ndarray,
        old_features: np.ndarray,
    ) -> tuple[float, float]:
        if new_features is None or len(new_features) == 0:
            return 0.0, 0.0

        valid_new = new_features[status.ravel() == 1]
        valid_old = old_features[status.ravel() == 1]

        if len(valid_new) <= 5:
            return 0.0, 0.0

        flow = valid_new.reshape(-1, 2) - valid_old.reshape(-1, 2)
        return np.median(flow[:, 0]), np.median(flow[:, 1])  # dx, dy in pixels
