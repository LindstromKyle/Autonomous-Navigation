import cv2
import numpy as np
from autonomous_nav.config import AppConfig
from autonomous_nav.mission_manager import MissionMode


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
        current_mode: MissionMode,
    ) -> np.ndarray:
        img = frame.copy()
        h, w = frame.shape[:2]

        # Draw optical flow trails and current feature points
        for new, old in zip(new_pts, old_pts):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            trail_mask = cv2.line(trail_mask, (a, b), (c, d), (0, 255, 200), 2)
            img = cv2.circle(img, (a, b), 5, (0, 0, 255), -1)

        img = cv2.add(img, trail_mask)

        # === Hazard Grid: Red borders always drawn ===
        gs = self.config.hazard.grid_size
        ch, cw = h // gs, w // gs

        # Temporary overlay for semi-transparent green safe fills
        overlay = img.copy()

        # Always draw red grid borders on all cells
        for r in range(gs):
            for c in range(gs):
                tl = (c * cw, r * ch)
                br = ((c + 1) * cw, (r + 1) * ch)
                cv2.rectangle(img, tl, br, (0, 0, 255), 2)

        # Only fill safe cells with green during landing phases
        if current_mode in (
            MissionMode.LANDING_APPROACH,
            MissionMode.LANDED_SAFE,
            MissionMode.NO_SAFE_ZONE,
        ):
            for r in range(gs):
                for c in range(gs):
                    if not hazard_mask[r, c]:
                        tl_inset = (c * cw + 3, r * ch + 3)
                        br_inset = ((c + 1) * cw - 3, (r + 1) * ch - 3)
                        cv2.rectangle(overlay, tl_inset, br_inset, (0, 255, 0), -1)

            # Blend green fills with lower alpha (more transparent)
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

            # Bright green borders on safe cells
            for r in range(gs):
                for c in range(gs):
                    if not hazard_mask[r, c]:
                        tl = (c * cw, r * ch)
                        br = ((c + 1) * cw, (r + 1) * ch)
                        cv2.rectangle(img, tl, br, (0, 255, 0), 3)

        # === Precise selected landing site marker — ONLY in landing phases ===
        if (
            current_mode
            in (
                MissionMode.LANDING_APPROACH,
                MissionMode.LANDED_SAFE,
                MissionMode.NO_SAFE_ZONE,
            )
            and safe_center_px is not None
        ):
            cx, cy = safe_center_px
            cv2.circle(img, (cx, cy), 45, (0, 255, 0), 5)
            cv2.circle(img, (cx, cy), 30, (0, 220, 0), -1)  # Solid inner circle
            cv2.putText(
                img,
                "SAFE LANDING SITE",
                (cx - 85, cy - 55),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # === Mode Indicator Text (top center) ===
        if current_mode == MissionMode.NAVIGATION:
            mode_text = "NAVIGATION MODE"
            mode_color = (255, 200, 0)
        elif current_mode == MissionMode.LANDING_APPROACH:
            mode_text = "LANDING APPROACH"
            mode_color = (0, 255, 255)
        elif current_mode == MissionMode.LANDED_SAFE:
            mode_text = "LANDED - SAFE"
            mode_color = (0, 255, 0)
        else:  # NO_SAFE_ZONE
            mode_text = "NO SAFE ZONE"
            mode_color = (0, 0, 255)

        text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 30
        cv2.rectangle(
            img,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            mode_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            mode_color,
            3,
        )

        # === Mode-specific additional visuals and text ===
        center = (w // 2, h // 2)
        remaining_dist_cm = np.hypot(remaining_x_cm, remaining_y_cm)
        ppc = self.config.global_.pixels_per_cm

        if (
            current_mode == MissionMode.NAVIGATION
            and remaining_dist_cm > self.config.navigation.arrival_threshold_cm
        ):
            # Arrow to original mission target
            direction = np.array([remaining_x_cm, -remaining_y_cm])
            direction /= remaining_dist_cm
            max_len = self.config.navigation.arrow_max_length_px
            arrow_length_px = min(remaining_dist_cm * ppc, max_len)
            end_point = center + (direction * arrow_length_px).astype(int)
            end_point = (int(end_point[0]), int(end_point[1]))
            cv2.arrowedLine(img, center, end_point, (255, 200, 0), 6, tipLength=0.3)

            cv2.putText(
                img,
                f"Target: {remaining_dist_cm:.1f} cm",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2,
            )

        elif current_mode == MissionMode.LANDING_APPROACH:
            if safe_center_px is not None:
                cv2.putText(
                    img,
                    "APPROACHING SAFE SITE",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    img,
                    "SEARCHING FOR SAFE SITE",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                )

        elif current_mode == MissionMode.LANDED_SAFE:
            cv2.putText(
                img,
                "MISSION COMPLETE",
                (w // 2 - 180, h // 2 + 20),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2,
                (0, 255, 0),
                4,
            )

        elif current_mode == MissionMode.NO_SAFE_ZONE:
            warning_text = "NO SAFE LANDING ZONE FOUND"
            text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 3)[
                0
            ]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2 + 20
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
                1.0,
                (255, 255, 255),
                3,
            )

        # === Standard info overlays ===
        cv2.putText(
            img,
            f"Pos: ({pos_x:+.1f}, {pos_y:+.1f}) cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Features: {num_features}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            f"Hazards: {np.sum(hazard_mask)}/{gs**2}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "r = reset pos | q = quit",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        return img
