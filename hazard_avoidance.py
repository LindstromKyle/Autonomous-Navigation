import numpy as np
from scipy.ndimage import label, find_objects
from abc import ABC, abstractmethod
from config import AppConfig, HazardConfig
from utils import pixels_to_cm


class HazardAvoidance(ABC):
    @abstractmethod
    def detect(
        self, features: np.ndarray, frame_height: int, frame_width: int
    ) -> tuple[float | None, float | None, np.ndarray, tuple[int, int] | None]:
        """
        Returns:
            dx_cm, dy_cm to safe zone,
            hazard_mask (bool grid),
            safe_center_px (for visualization)
        """
        pass


class DensityBasedHazardAvoidance(HazardAvoidance):
    def __init__(self, config: HazardConfig, global_config: AppConfig):
        self.config = config
        self.pixels_per_cm = global_config.global_.pixels_per_cm

    def detect(
        self, features: np.ndarray, frame_height: int, frame_width: int
    ) -> tuple[float | None, float | None, np.ndarray, tuple[int, int] | None]:
        grid_size = self.config.grid_size
        cell_h = frame_height // grid_size
        cell_w = frame_width // grid_size
        density = np.zeros((grid_size, grid_size), dtype=int)

        if len(features) == 0:
            hazard_mask = density > self.config.threshold
            return None, None, hazard_mask, None

        for x, y in features:
            gx = min(int(x // cell_w), grid_size - 1)
            gy = min(int(y // cell_h), grid_size - 1)
            density[gy, gx] += 1

        hazard_mask = density > self.config.threshold

        if self.config.exclude_boundaries and grid_size > 2:
            hazard_mask[0, :] = True
            hazard_mask[-1, :] = True
            hazard_mask[:, 0] = True
            hazard_mask[:, -1] = True

        safe_mask = ~hazard_mask
        if not np.any(safe_mask):
            return None, None, hazard_mask, None

        labeled, num = label(safe_mask)
        sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
        largest_id = np.argmax(sizes) + 1
        safe_slice = find_objects(labeled == largest_id)[0]

        # Geometric center of largest safe zone
        cy = (safe_slice[0].start + safe_slice[0].stop - 1) // 2
        cx = (safe_slice[1].start + safe_slice[1].stop - 1) // 2
        geo_px = (cx * cell_w + cell_w // 2, cy * cell_h + cell_h // 2)

        # Closest actual safe cell center
        safe_ys, safe_xs = np.where(labeled == largest_id)
        cell_centers = [
            (sx * cell_w + cell_w // 2, sy * cell_h + cell_h // 2)
            for sy, sx in zip(safe_ys, safe_xs)
        ]
        distances = [
            np.hypot(px - geo_px[0], py - geo_px[1]) for px, py in cell_centers
        ]
        closest = cell_centers[np.argmin(distances)]

        dx_px = closest[0] - frame_width // 2
        dy_px = closest[1] - frame_height // 2
        dx_cm = pixels_to_cm(dx_px, self.pixels_per_cm)
        dy_cm = pixels_to_cm(dy_px, self.pixels_per_cm)

        return dx_cm, dy_cm, hazard_mask, closest
