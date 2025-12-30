import numpy as np
import cv2
from scipy.ndimage import label, find_objects
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
    ) -> tuple[float | None, float | None, np.ndarray, tuple[int, int] | None]:
        """
        Returns:
            dx_cm, dy_cm to safe zone,
            hazard_mask (bool grid),
            safe_center_px (for visualization)
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
    ) -> tuple[float | None, float | None, np.ndarray, tuple[int, int] | None]:
        if len(features) == 0:
            # No features: Assume fully safe, center is safe
            safe_center_px = (frame_width // 2, frame_height // 2)
            dx_cm = dy_cm = 0.0
            hazard_mask = np.zeros(
                (self.config.grid_size, self.config.grid_size), dtype=bool
            )
            return dx_cm, dy_cm, hazard_mask, safe_center_px

        # Create binary hazard mask: Mark features as hazards (0 = hazard, 255 = safe initially)
        hazard_img = np.full((frame_height, frame_width), 255, dtype=np.uint8)
        for x, y in features.astype(int):
            cv2.circle(
                hazard_img, (x, y), self.config.hazard_dilation_px, 0, -1
            )  # Dilate for radius

        # Distance transform: Distance to nearest hazard (higher = safer)
        dist_map = cv2.distanceTransform(hazard_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Mask out areas below min clearance
        dist_map[dist_map < self.min_clearance_px] = 0

        if np.max(dist_map) == 0:
            # No safe areas
            return (
                None,
                None,
                np.ones((self.config.grid_size, self.config.grid_size), dtype=bool),
                None,
            )

        # Bias toward target if in landing mode
        weighted_map = dist_map.copy().astype(float)
        if target_px is not None:
            # Create Gaussian proximity map (higher near target, inverted distance)
            gx, gy = np.meshgrid(np.arange(frame_width), np.arange(frame_height))
            prox_map = np.exp(
                -((gx - target_px[0]) ** 2 + (gy - target_px[1]) ** 2)
                / (2 * (frame_width / 4) ** 2)
            )
            weighted_map += (
                self.config.proximity_bias_weight * prox_map * np.max(dist_map)
            )

        # Find optimal safe point (argmax of weighted safety)
        safe_y, safe_x = np.unravel_index(np.argmax(weighted_map), weighted_map.shape)
        safe_center_px = (safe_x, safe_y)

        # Compute relative offsets
        center_x, center_y = frame_width // 2, frame_height // 2
        dx_px = safe_x - center_x
        dy_px = safe_y - center_y
        dx_cm = pixels_to_cm(dx_px, self.pixels_per_cm)
        dy_cm = pixels_to_cm(dy_px, self.pixels_per_cm)

        # Generate coarse hazard_mask for visualization (fall back to grid)
        gs = self.config.grid_size
        ch, cw = frame_height // gs, frame_width // gs
        hazard_mask = np.zeros((gs, gs), dtype=bool)
        for r in range(gs):
            for c in range(gs):
                cell_slice = dist_map[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw]
                if (
                    np.mean(cell_slice) < self.min_clearance_px
                ):  # Average clearance low = hazard
                    hazard_mask[r, c] = True

        if self.config.exclude_boundaries and gs > 2:
            hazard_mask[0, :] = hazard_mask[-1, :] = hazard_mask[:, 0] = hazard_mask[
                :, -1
            ] = True

        return dx_cm, dy_cm, hazard_mask, safe_center_px


class DensityBasedHazardAvoidance(HazardAvoidance):
    def __init__(self, config: HazardConfig, global_config: AppConfig):
        self.config = config
        self.pixels_per_cm = global_config.global_.pixels_per_cm

    def detect(
        self,
        features: np.ndarray,
        frame_height: int,
        frame_width: int,
        target_px: tuple[int, int] | None = None,
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
        if num == 0:
            return None, None, hazard_mask, None

        cell_centers = []
        sizes = []
        for i in range(1, num + 1):
            safe_slice = find_objects(labeled == i)[0]
            cy = (safe_slice[0].start + safe_slice[0].stop - 1) // 2
            cx = (safe_slice[1].start + safe_slice[1].stop - 1) // 2
            center_px = (cx * cell_w + cell_w // 2, cy * cell_h + cell_h // 2)
            size = np.sum(labeled == i)
            cell_centers.append(center_px)
            sizes.append(size)

        if target_px is None:
            # Standard mode: largest safe zone (always return one if any exist)
            if num > 0:
                best_idx = np.argmax(sizes)
                safe_center_px = cell_centers[best_idx]
            else:
                safe_center_px = None
        else:
            # Landing mode: only accept zones close to target_px
            distances = [
                np.hypot(cx - target_px[0], cy - target_px[1])
                for cx, cy in cell_centers
            ]
            proximity_threshold_px = min(frame_width, frame_height) // 2

            candidates = [
                i for i, d in enumerate(distances) if d < proximity_threshold_px
            ]

            if candidates:
                # Pick largest among proximate ones
                best_idx = candidates[np.argmax([sizes[i] for i in candidates])]
                safe_center_px = cell_centers[best_idx]
            else:
                # No safe zone near target → explicitly return None
                safe_center_px = None

        if safe_center_px is None:
            dx_cm = dy_cm = None
        else:
            dx_px = safe_center_px[0] - frame_width // 2
            dy_px = safe_center_px[1] - frame_height // 2
            dx_cm = pixels_to_cm(dx_px, self.pixels_per_cm)
            dy_cm = pixels_to_cm(dy_px, self.pixels_per_cm)

        return dx_cm, dy_cm, hazard_mask, safe_center_px
