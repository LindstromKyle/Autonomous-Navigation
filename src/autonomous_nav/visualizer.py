import cv2
import numpy as np
from autonomous_nav.config import AppConfig


class Visualizer:
    def __init__(self, config: AppConfig):
        self.config = config

    def annotate_frame(
        self,
        frame: np.ndarray,
        trail_mask: np.ndarray,
        new_pts: np.ndarray,
        old_pts: np.ndarray,
        pos_x: float,
        pos_y: float,
        num_features: int,
        hazard_mask: np.ndarray,
        safe_center_px: tuple[int, int] | None,
    ) -> np.ndarray:
        # Draw trails and points
        for new, old in zip(new_pts, old_pts):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            trail_mask = cv2.line(trail_mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

        img = cv2.add(frame, trail_mask)

        # Hazard grid
        h, w = frame.shape[:2]
        gs = self.config.hazard.grid_size
        ch, cw = h // gs, w // gs
        for r in range(gs):
            for c in range(gs):
                if hazard_mask[r, c]:
                    tl = (c * cw, r * ch)
                    br = ((c + 1) * cw, (r + 1) * ch)
                    cv2.rectangle(img, tl, br, (0, 0, 255), 2)

        if safe_center_px:
            cv2.circle(img, safe_center_px, 30, (0, 255, 0), 3)

        # Text overlays
        cv2.putText(
            img,
            f"Pos: ({pos_x:+.1f}, {pos_y:+.1f}) cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Features: {num_features}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Hazards: {np.sum(hazard_mask)}/{gs**2}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "r = reset pos | q = quit",
            (10, img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        return img
