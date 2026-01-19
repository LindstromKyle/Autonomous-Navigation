import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
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
        state: np.ndarray,
        num_features: int,
        dx_to_search_zone_cm: float,
        dy_to_search_zone_cm: float,
        dx_to_locked_landing_target_cm: float | None,
        dy_to_locked_landing_target_cm: float | None,
        mission_manager,
        weighted_map: np.ndarray | None = None,
        hazard_points: np.ndarray | None = None,
    ) -> np.ndarray:
        current_mode = mission_manager.current_mode

        img = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        center = np.array([cx, cy])

        pos_x = state.position[0]
        pos_y = state.position[1]
        pos_z = state.position[2]

        # Draw points
        for pt in new_pts:
            x, y = map(int, pt.ravel())
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        # Draw new trails
        for new, old in zip(new_pts, old_pts):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            cv2.line(trail_mask, (a, b), (c, d), (0, 255, 200), 2)

        # Blend
        img = cv2.addWeighted(img, 1.0, trail_mask, 0.15, 0)

        if hazard_points is not None and len(hazard_points) > 0:
            for pt in hazard_points:
                x, y = map(int, pt.ravel())
                cv2.putText(
                    img,
                    "X",
                    (x - 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        overlay = img.copy()
        dist_to_search_zone_cm = np.hypot(dx_to_search_zone_cm, dy_to_search_zone_cm)

        # Navigation arrow (only in pure navigation mode)
        if current_mode == MissionMode.NAVIGATION:
            direction = np.array([dx_to_search_zone_cm, -dy_to_search_zone_cm])
            direction /= dist_to_search_zone_cm
            max_len = self.config.navigation.arrow_max_length_px
            arrow_length_px = min(dist_to_search_zone_cm * self.ppc, max_len)
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
                f"Target: {dist_to_search_zone_cm:.1f} cm",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2,
            )

        # === Heatmap + Search Zone (only in landing phases) ===
        else:
            # Re-compute original target projection here (safe, inside this block)
            orig_target_x_cm = self.config.navigation.planned_route_dx
            orig_target_y_cm = self.config.navigation.planned_route_dy
            orig_remaining_x = orig_target_x_cm - pos_x
            orig_remaining_y = orig_target_y_cm - pos_y

            target_px_x = w // 2 + int(orig_remaining_x * self.ppc)
            target_px_y = h // 2 - int(orig_remaining_y * self.ppc)
            target_px = (target_px_x, target_px_y)

            outer_radius_px = int(
                self.config.navigation.search_zone_outer_thresh * self.ppc
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

            if current_mode == MissionMode.SEARCHING:
                header_text = "SEARCHING"
                header_color = (0, 165, 255)
                cv2.putText(
                    img,
                    header_text,
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    header_color,
                    2,
                )

            elif current_mode == MissionMode.LANDING_APPROACH:
                landing_site_pixel = (
                    center
                    + np.array(
                        [
                            dx_to_locked_landing_target_cm,
                            -dy_to_locked_landing_target_cm,
                        ]
                    )
                    * self.ppc
                ).astype("int")
                header_text = "APPROACHING SAFE SITE"
                header_color = (0, 255, 255)
                cv2.putText(
                    img,
                    header_text,
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    header_color,
                    2,
                )
                cv2.circle(
                    img,
                    landing_site_pixel,
                    int(self.config.navigation.pos_tolerance_cm * self.ppc),
                    (0, 255, 0),
                    4,
                )
                dist_to_locked_landing_target_cm = np.hypot(
                    dx_to_locked_landing_target_cm, dy_to_locked_landing_target_cm
                )

                cv2.putText(
                    img,
                    f"Safe Site: {dist_to_locked_landing_target_cm:.1f} cm",
                    (10, 150),  # Below header
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            elif current_mode == MissionMode.HOVERING:
                header_text = "HOVERING"
                header_color = (0, 255, 0)
                cv2.putText(
                    img,
                    header_text,
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    header_color,
                    2,
                )
                landing_site_pixel = (
                    center
                    + np.array(
                        [
                            dx_to_locked_landing_target_cm,
                            -dy_to_locked_landing_target_cm,
                        ]
                    )
                    * self.ppc
                ).astype("int")
                # Show countdown instead of arrow
                hover_remaining = mission_manager.get_hover_remaining_s()
                cv2.putText(
                    img,
                    f"{hover_remaining:.1f}",
                    landing_site_pixel,  # Below header
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.circle(
                    img,
                    landing_site_pixel,
                    int(self.config.navigation.pos_tolerance_cm * self.ppc),
                    (0, 255, 0),
                    4,
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
                text_size = cv2.getTextSize(
                    warning_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 3
                )[0]
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
            f"Pos: ({pos_x:+.1f}, {pos_y:+.1f}, {pos_z:.1f}) cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        # Add orientation display: Convert quaternion to Euler angles (roll, pitch, yaw in degrees)
        # q = state.state[6:10]  # [q_w, q_x, q_y, q_z]
        # rot = R.from_quat([q[1], q[2], q[3], q[0]])  # SciPy expects [x, y, z, w]
        # euler_deg = rot.as_euler("xyz", degrees=True)  # Roll (x), Pitch (y), Yaw (z)
        # cv2.putText(
        #     img,
        #     f"Att: R:{euler_deg[0]:+.1f} P:{euler_deg[1]:+.1f} Y:{euler_deg[2]:+.1f} deg",
        #     (10, 60),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.6,
        #     (255, 255, 255),
        #     2,
        # )
        # print(
        #     f"DEBUG - EULER = R:{euler_deg[0]:+.1f} P:{euler_deg[1]:+.1f} Y:{euler_deg[2]:+.1f} deg"
        # )
        cv2.putText(
            img,
            f"Features: {num_features}",
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
        # Draw center indicator
        size = 20
        cv2.line(img, (cx - size, cy), (cx + size, cy), (255, 255, 255), 2)
        cv2.line(img, (cx, cy - size), (cx, cy + size), (255, 255, 255), 2)
        return img
