import numpy as np
import cv2
from abc import ABC, abstractmethod
from autonomous_nav.config import AppConfig, HazardConfig
from autonomous_nav.utils import pixels_to_cm


class HazardAvoidance(ABC):
    @abstractmethod
    def detect(
        self,
        features: np.ndarray,
        frame_height: int,
        frame_width: int,
        target_px: tuple[int, int] | None = None,
        outer_radius_cm: float | None = None,
    ) -> tuple[
        float | None, float | None, np.ndarray, tuple[int, int] | None, np.ndarray
    ]:
        """
        Returns:
            dx_cm, dy_cm to safe zone (relative to frame center),
            hazard_mask (bool grid),
            safe_center_px (for visualization and mission manager),
            weighted_map (full-resolution float safety score map for heatmap)
        """
        pass


class ClearanceBasedHazardAvoidance(HazardAvoidance):
    def __init__(self, config: HazardConfig, global_config: AppConfig):
        self.config = config
        self.pixels_per_cm = global_config.global_.pixels_per_cm
        self.min_clearance_px = int(self.config.min_clearance_cm * self.pixels_per_cm)

    def detect(
        self,
        features: np.ndarray,
        frame_height: int,
        frame_width: int,
        target_px: tuple[int, int] | None = None,
        outer_radius_cm: float | None = None,
    ) -> tuple[
        float | None, float | None, np.ndarray, tuple[int, int] | None, np.ndarray
    ]:
        gs = self.config.grid_size
        ch, cw = frame_height // gs, frame_width // gs

        # Create binary hazard image: 255 = safe, 0 = hazard
        hazard_img = np.full((frame_height, frame_width), 255, dtype=np.uint8)

        # Dilate features to represent hazard radius
        for x, y in features.astype(int):
            cv2.circle(
                hazard_img,
                (x, y),
                self.config.hazard_dilation_px,
                0,
                -1,
            )

        # Distance transform: higher = safer
        dist_map = cv2.distanceTransform(hazard_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Enforce minimum clearance
        dist_map[dist_map < self.min_clearance_px] = 0

        # Search mask for landing phase
        search_mask = np.ones((frame_height, frame_width), dtype=bool)
        if outer_radius_cm is not None and target_px is not None:
            outer_radius_px = int(outer_radius_cm * self.pixels_per_cm + 0.5)
            cy, cx = np.ogrid[:frame_height, :frame_width]
            dist_from_target_sq = (cx - target_px[0]) ** 2 + (cy - target_px[1]) ** 2
            search_mask = dist_from_target_sq <= outer_radius_px**2

        # Weighted map is now purely clearance-based
        weighted_map = dist_map.astype(float)
        weighted_map[~search_mask] = 0  # Zero outside search zone

        # Find safest pixel
        if np.max(weighted_map) > 0:
            safe_y, safe_x = np.unravel_index(
                np.argmax(weighted_map), weighted_map.shape
            )
            safe_center_px = (int(safe_x), int(safe_y))
            dx_px = safe_center_px[0] - frame_width // 2
            dy_px = safe_center_px[1] - frame_height // 2
            dx_cm = pixels_to_cm(dx_px, self.pixels_per_cm)
            dy_cm = pixels_to_cm(dy_px, self.pixels_per_cm)
        else:
            safe_center_px = None
            dx_cm = dy_cm = None

        # Grid-based hazard mask for stats/visualization
        hazard_mask = np.zeros((gs, gs), dtype=bool)
        for r in range(gs):
            for c in range(gs):
                y_start, y_end = r * ch, (r + 1) * ch
                x_start, x_end = c * cw, (c + 1) * cw
                cell_slice = dist_map[y_start:y_end, x_start:x_end]
                if np.mean(cell_slice) < self.min_clearance_px:
                    hazard_mask[r, c] = True

        if self.config.exclude_boundaries and gs > 2:
            hazard_mask[0, :] = hazard_mask[-1, :] = hazard_mask[:, 0] = hazard_mask[
                :, -1
            ] = True

        return dx_cm, dy_cm, hazard_mask, safe_center_px, weighted_map
