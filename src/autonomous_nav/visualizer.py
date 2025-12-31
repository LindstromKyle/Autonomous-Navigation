import cv2
import numpy as np
from autonomous_nav.config import AppConfig
from autonomous_nav.mission_manager import MissionMode


class Visualizer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.ppc = config.global_.pixels_per_cm  # For consistent scaling

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
        weighted_map: np.ndarray | None = None,
    ) -> np.ndarray:
        img = frame.copy()
        h, w = frame.shape[:2]
        center = np.array([w // 2, h // 2])

        # Draw optical flow trails and points
        for new, old in zip(new_pts, old_pts):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            trail_mask = cv2.line(trail_mask, (a, b), (c, d), (0, 255, 200), 2)
            img = cv2.circle(img, (a, b), 5, (0, 0, 255), -1)
        img = cv2.add(img, trail_mask)

        overlay = img.copy()

        # === Heatmap + Search Zone (only in landing phases) ===
        if current_mode in (
            MissionMode.LANDING_APPROACH,
            MissionMode.LANDED_SAFE,
            MissionMode.NO_SAFE_ZONE,
        ):
            # Re-compute original target projection here (safe, inside this block)
            orig_target_x_cm = self.config.navigation.target_offset_x_cm
            orig_target_y_cm = self.config.navigation.target_offset_y_cm
            orig_remaining_x = orig_target_x_cm - pos_x
            orig_remaining_y = orig_target_y_cm - pos_y

            target_px_x = w // 2 + int(orig_remaining_x * self.ppc)
            target_px_y = h // 2 - int(orig_remaining_y * self.ppc)
            target_px = (target_px_x, target_px_y)

            outer_radius_px = int(
                self.config.navigation.arrival_outer_threshold_cm * self.ppc
            )

            if weighted_map is not None:
                if np.max(weighted_map) > 0:
                    min_nonzero = np.min(weighted_map[weighted_map > 0])
                    # Set zeros to just below min_nonzero (e.g., 99% of it for a small delta)
                    weighted_map[weighted_map == 0] = min_nonzero * 0.99
                    # If all zero, heatmap will normalize to flat zero (no contrast, which is fine)

                norm_map = cv2.normalize(weighted_map, None, 0, 255, cv2.NORM_MINMAX)
                norm_map = norm_map.astype(np.uint8)
                heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_VIRIDIS)
                alpha = 0.35
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, target_px, outer_radius_px, 255, -1)
                heatmap_masked = cv2.bitwise_and(heatmap, heatmap, mask=mask)

                cv2.addWeighted(heatmap_masked, alpha, overlay, 1 - alpha, 0, overlay)

            # Faint grid (unchanged)
            gs = self.config.hazard.grid_size
            ch, cw = h // gs, w // gs
            for r in range(gs):
                for c in range(gs):
                    tl = (c * cw, r * ch)
                    br = ((c + 1) * cw, (r + 1) * ch)
                    cv2.rectangle(overlay, tl, br, (50, 50, 50), 1)

            img = overlay

            # Search radius circle (centered on original target)
            cv2.circle(img, target_px, outer_radius_px, (0, 255, 0), 4)

            # Safe site marker (already inside landing phase block)
            if safe_center_px is not None:
                cv2.drawMarker(
                    img, safe_center_px, (0, 255, 0), cv2.MARKER_CROSS, 40, 1
                )

        remaining_dist_cm = np.hypot(remaining_x_cm, remaining_y_cm)

        # Navigation arrow (only in pure navigation mode)
        if (
            current_mode == MissionMode.NAVIGATION
            and remaining_dist_cm > self.config.navigation.arrival_inner_threshold_cm
        ):
            direction = np.array([remaining_x_cm, -remaining_y_cm])
            direction /= remaining_dist_cm
            max_len = self.config.navigation.arrow_max_length_px
            arrow_length_px = min(remaining_dist_cm * self.ppc, max_len)
            end_point = center + (direction * arrow_length_px).astype(int)
            cv2.arrowedLine(
                img,
                tuple(center.astype(int)),
                tuple(map(int, end_point)),
                (255, 200, 0),
                6,
                tipLength=0.3,
            )
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

        # Standard info
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
            f"Hazards: {np.sum(hazard_mask)}/{self.config.hazard.grid_size**2}",
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
