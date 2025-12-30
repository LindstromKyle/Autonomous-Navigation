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
        remaining_x_cm: float,
        remaining_y_cm: float,
        in_landing_mode: bool,
    ) -> np.ndarray:
        img = frame.copy()

        # Draw trails and points
        for new, old in zip(new_pts, old_pts):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            trail_mask = cv2.line(trail_mask, (a, b), (c, d), (0, 255, 200), 2)
            img = cv2.circle(img, (a, b), 5, (0, 0, 255), -1)

        img = cv2.add(img, trail_mask)

        # === Hazard Grid: Red borders always, Green infill ONLY in landing mode ===
        h, w = frame.shape[:2]
        gs = self.config.hazard.grid_size
        ch, cw = h // gs, w // gs

        # Temporary overlay for green safe fills
        overlay = img.copy()

        # Always draw red grid borders on all cells
        for r in range(gs):
            for c in range(gs):
                tl = (c * cw, r * ch)
                br = ((c + 1) * cw, (r + 1) * ch)
                cv2.rectangle(img, tl, br, (0, 0, 255), 2)

        # Only fill and highlight safe cells when in landing mode
        if in_landing_mode:
            # Semi-transparent green fill on safe cells
            for r in range(gs):
                for c in range(gs):
                    if not hazard_mask[r, c]:
                        tl_inset = (c * cw + 3, r * ch + 3)
                        br_inset = ((c + 1) * cw - 3, (r + 1) * ch - 3)
                        cv2.rectangle(overlay, tl_inset, br_inset, (0, 255, 0), -1)

            # Blend green fills
            img = cv2.addWeighted(overlay, 0.4, img, 0.5, 0)

            # Bright green borders on safe cells
            for r in range(gs):
                for c in range(gs):
                    if not hazard_mask[r, c]:
                        tl = (c * cw, r * ch)
                        br = ((c + 1) * cw, (r + 1) * ch)
                        cv2.rectangle(img, tl, br, (0, 255, 0), 3)

        # === Mode Indicator Text (top center) ===
        mode_text = "LANDING MODE" if in_landing_mode else "NAVIGATION MODE"
        mode_color = (0, 255, 0) if in_landing_mode else (255, 200, 0)
        text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 30

        # Background strip
        cv2.rectangle(
            img,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 0, 0),
            -1,
        )
        # Text
        cv2.putText(
            img,
            mode_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            mode_color,
            2,
        )

        # === GUIDANCE CUES ===
        if not in_landing_mode:
            # Navigation phase: Blue arrow
            if abs(remaining_x_cm) > 0.1 or abs(remaining_y_cm) > 0.1:
                center = (w // 2, h // 2)
                remaining_dist_cm = np.hypot(remaining_x_cm, remaining_y_cm)
                ppc = self.config.global_.pixels_per_cm
                direction = np.array([remaining_x_cm, -remaining_y_cm])
                direction /= remaining_dist_cm
                max_len = getattr(self.config.navigation, "arrow_max_length_px", 200)
                arrow_length_px = min(remaining_dist_cm * ppc, max_len)
                end_point = center + (direction * arrow_length_px).astype(int)
                end_point = (int(end_point[0]), int(end_point[1]))
                cv2.arrowedLine(img, center, end_point, (255, 200, 0), 6, tipLength=0.3)

                cv2.putText(
                    img,
                    f"Target: {remaining_dist_cm:.1f} cm",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 200, 0),
                    2,
                )
        else:
            # Landing phase
            if safe_center_px is not None:
                cv2.circle(img, safe_center_px, 30, (0, 255, 0), 4)
                cv2.circle(img, safe_center_px, 40, (0, 255, 0), 3)
            else:
                warning_text = "NO SAFE LANDING ZONE"
                text_size = cv2.getTextSize(
                    warning_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 4
                )[0]
                text_x = (img.shape[1] - text_size[0]) // 2
                text_y = img.shape[0] // 2

                cv2.rectangle(
                    img,
                    (text_x - 20, text_y - text_size[1] - 20),
                    (text_x + text_size[0] + 20, text_y + 20),
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    img,
                    warning_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.5,
                    (255, 255, 255),
                    4,
                )

        # Standard text overlays
        cv2.putText(
            img,
            f"Pos: ({pos_x:+.1f}, {pos_y:+.1f}) cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Features: {num_features}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Hazards: {np.sum(hazard_mask)}/{gs**2}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "r = reset pos | q = quit",
            (10, img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            2,
        )

        return img
