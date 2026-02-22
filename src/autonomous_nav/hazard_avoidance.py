import numpy as np
import cv2
from abc import ABC, abstractmethod
from autonomous_nav.config import AppConfig, HazardConfig
from autonomous_nav.utils import cm_to_pixels, pixels_to_cm
from ultralytics import YOLO


class HazardAvoidance(ABC):
    """
    ABC for hazard avoidance module
    """

    def __init__(self, config: HazardConfig, global_config: AppConfig):
        self.config = config
        self.global_config = global_config.global_
        self.hazard_points: np.ndarray | None = None

    @abstractmethod
    def compute_hazards(
        self,
        features: np.ndarray | None = None,
        frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Subclasses must implement this method
        """
        pass

    def compute_safe_zone(
        self,
        features: np.ndarray,
        frame_height: int,
        frame_width: int,
        current_height: float,
        search_zone_center: tuple[int, int] | None = None,
        search_zone_outer_thresh: float | None = None,
        frame: np.ndarray | None = None,
    ) -> tuple[
        float | None, float | None, np.ndarray, tuple[int, int] | None, np.ndarray
    ]:
        """
        Computes a safe zone based on hazards in the input image
        """

        # Generate hazard points based on subclass
        self.hazard_points = self.compute_hazards(
            features=features.reshape(-1, 2) if features.size > 0 else None,
            frame=frame,
        )

        # If no hazards detected, treat entire frame as safe
        if self.hazard_points.size == 0:
            self.hazard_points = np.empty((0, 2), dtype=np.float32)

        # Binary hazard image
        hazard_img = np.full((frame_height, frame_width), 255, dtype=np.uint8)

        # Dilate hazard points
        for x, y in self.hazard_points.astype(int):
            cv2.circle(hazard_img, (x, y), self.config.hazard_dilation_px, 0, -1)

        # Distance transform
        dist_map = cv2.distanceTransform(hazard_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        min_clearance = cm_to_pixels(
            self.config.min_clearance_cm,
            current_height,
            self.global_config.focal_length_px,
        )
        dist_map[dist_map < min_clearance] = 0

        # Search zone mask
        search_mask = np.ones((frame_height, frame_width), dtype=bool)
        if search_zone_outer_thresh is not None and search_zone_center is not None:
            outer_radius_px = int(
                cm_to_pixels(
                    search_zone_outer_thresh,
                    current_height,
                    self.global_config.focal_length_px,
                )
                + 0.5
            )
            cy, cx = np.ogrid[:frame_height, :frame_width]
            dist_sq = (cx - search_zone_center[0]) ** 2 + (
                cy - search_zone_center[1]
            ) ** 2
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
            dx_cm = pixels_to_cm(
                dx_px, current_height, self.global_config.focal_length_px
            )
            dy_cm = -pixels_to_cm(
                dy_px, current_height, self.global_config.focal_length_px
            )
        else:
            safe_center_px = None
            dx_cm = dy_cm = None

        return dx_cm, dy_cm, safe_center_px, weighted_map


class ClearanceBasedHazardAvoidance(HazardAvoidance):
    """
    This class simple re-uses the corners from the Shi-Tomasi feature detector as
    landing site hazards
    """

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
    """
    This class runs an AI based neural network against the input frame to determine
    landing site hazards
    """

    def __init__(self, config: HazardConfig, global_config: AppConfig):
        super().__init__(config, global_config)
        # Set up network
        self.model = YOLO(
            "/home/kyle/repos/Autonomous-Navigation/examples/AI/mars_rocks_custom.pt"
        )
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

    def compute_hazards(
        self,
        features: np.ndarray | None = None,
        frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Runs the network on the input image
        """
        if frame is None:
            raise ValueError("AIHazardAvoidance requires the RGB frame")

        # Run network
        results = self.model(frame, imgsz=320, conf=0.15, verbose=False)[0]

        # Grab centroids from bboxes
        centroids = []
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                centroids.append([(x1 + x2) / 2, (y1 + y2) / 2])

        return (
            np.array(centroids, dtype=np.float32)
            if centroids
            else np.empty((0, 2), dtype=np.float32)
        )
