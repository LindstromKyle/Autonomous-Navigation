import numpy as np
import cv2
from abc import ABC, abstractmethod
from autonomous_nav.config import AppConfig, HazardConfig
from autonomous_nav.utils import pixels_to_cm
from ultralytics import YOLO


class HazardAvoidance(ABC):
    def __init__(self, config: HazardConfig, global_config: AppConfig):
        self.config = config
        self.global_config = global_config
        self.pixels_per_cm = global_config.global_.pixels_per_cm
        self.min_clearance_px = int(self.config.min_clearance_cm * self.pixels_per_cm)
        self.hazard_points: np.ndarray | None = None

    @abstractmethod
    def compute_hazards(
        self,
        features: np.ndarray | None = None,
        frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Subclass implements this to produce hazard points (N, 2) float32 array.
        Returns empty array if no hazards.
        """
        pass

    def compute_safe_zone(
        self,
        features: np.ndarray,
        frame_height: int,
        frame_width: int,
        target_px: tuple[int, int] | None = None,
        outer_radius_cm: float | None = None,
        frame: np.ndarray | None = None,
    ) -> tuple[
        float | None, float | None, np.ndarray, tuple[int, int] | None, np.ndarray
    ]:
        """
        Main public method — delegates to compute_hazards then processes.
        """
        # Let subclass generate hazard points using whatever input it needs
        self.hazard_points = self.compute_hazards(
            features=features.reshape(-1, 2) if features.size > 0 else None,
            frame=frame,
        )

        # If no hazards detected, treat entire frame as safe (or handle as needed)
        if self.hazard_points.size == 0:
            self.hazard_points = np.empty((0, 2), dtype=np.float32)

        gs = self.config.grid_size
        ch, cw = frame_height // gs, frame_width // gs

        # Binary hazard image
        hazard_img = np.full((frame_height, frame_width), 255, dtype=np.uint8)

        # Dilate hazard points
        for x, y in self.hazard_points.astype(int):
            cv2.circle(hazard_img, (x, y), self.config.hazard_dilation_px, 0, -1)

        # Distance transform
        dist_map = cv2.distanceTransform(hazard_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_map[dist_map < self.min_clearance_px] = 0

        # Search zone mask
        search_mask = np.ones((frame_height, frame_width), dtype=bool)
        if outer_radius_cm is not None and target_px is not None:
            outer_radius_px = int(outer_radius_cm * self.pixels_per_cm + 0.5)
            cy, cx = np.ogrid[:frame_height, :frame_width]
            dist_sq = (cx - target_px[0]) ** 2 + (cy - target_px[1]) ** 2
            search_mask = dist_sq <= outer_radius_px**2

        # Weighted safety map
        weighted_map = dist_map.astype(float)
        weighted_map[~search_mask] = 0

        # Find safest point
        if np.max(weighted_map) > 0:
            safe_y, safe_x = np.unravel_index(
                np.argmax(weighted_map), weighted_map.shape
            )
            safe_center_px = (int(safe_x), int(safe_y))
            dx_px = safe_x - frame_width // 2
            dy_px = safe_y - frame_height // 2
            dx_cm = pixels_to_cm(dx_px, self.pixels_per_cm)
            dy_cm = pixels_to_cm(dy_px, self.pixels_per_cm)
        else:
            safe_center_px = None
            dx_cm = dy_cm = None

        # Grid hazard mask for visualization/stats
        hazard_mask = np.zeros((gs, gs), dtype=bool)
        for r in range(gs):
            for c in range(gs):
                y_start, y_end = r * ch, (r + 1) * ch
                x_start, x_end = c * cw, (c + 1) * cw
                if (
                    np.mean(dist_map[y_start:y_end, x_start:x_end])
                    < self.min_clearance_px
                ):
                    hazard_mask[r, c] = True

        if self.config.exclude_boundaries and gs > 2:
            hazard_mask[0, :] = hazard_mask[-1, :] = hazard_mask[:, 0] = hazard_mask[
                :, -1
            ] = True

        return dx_cm, dy_cm, hazard_mask, safe_center_px, weighted_map


class ClearanceBasedHazardAvoidance(HazardAvoidance):
    def compute_hazards(
        self,
        features: np.ndarray | None = None,
        frame: np.ndarray | None = None,
    ) -> np.ndarray:
        # Simply use the passed features as hazards
        if features is None or features.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return features.astype(np.float32)


class AIHazardAvoidance(HazardAvoidance):
    def __init__(self, config: HazardConfig, global_config: AppConfig):
        super().__init__(config, global_config)
        self.model = YOLO(
            "/home/kyle/repos/Autonomous-Navigation/examples/AI/full_marsdata_v2.pt"
        )
        # Warm-up
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

    def compute_hazards(
        self,
        features: np.ndarray | None = None,
        frame: np.ndarray | None = None,
    ) -> np.ndarray:
        if frame is None:
            raise ValueError("AIHazardAvoidance requires the RGB frame")

        results = self.model(frame, imgsz=320, conf=0.005, verbose=False)[0]

        centroids = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:  # Adjust if your rock class is different
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])

        return (
            np.array(centroids, dtype=np.float32)
            if centroids
            else np.empty((0, 2), dtype=np.float32)
        )
