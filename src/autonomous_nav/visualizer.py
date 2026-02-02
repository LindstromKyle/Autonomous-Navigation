import cv2
import numpy as np
from autonomous_nav.config import AppConfig
from autonomous_nav.mission_manager import MissionMode
from autonomous_nav.utils import cm_to_pixels


class Visualizer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.focal_px = config.global_.focal_length_px
        self.mode_colors = {
            "ascent": (255, 200, 0),  # blue
            "navigation": (255, 200, 0),  # blue
            "searching": (0, 255, 255),  # yellow
            "landing_approach": (0, 255, 255),  # yellow
            "descent": (0, 255, 0),  # green
            "hovering": (0, 255, 0),  # green
            "landed_safe": (0, 255, 0),  # green
            "no_safe_zone": (0, 0, 255),  # red
        }

    def mode_display_name(self, current_mode: MissionMode) -> str:
        return current_mode.value.replace("_", " ").upper()

    def draw_altitude_progress_bars(
        self,
        frame: np.ndarray,
        current_mode: MissionMode,
        current_mode_color: tuple,
        pos_z: float,
    ) -> None:
        """
        Draws a vertical progress bar on the left side:
        - ASCENT:   fills upward (bottom → top) toward target height
        - DESCENT:  empties downward (top → bottom) toward final height
        Hides in other modes.
        """

        bar_width = 15
        bar_height = 100
        x_left = 200
        y_top = 200
        bg_color: tuple = (40, 40, 40)
        border_color: tuple = (180, 180, 180)

        if current_mode not in (
            MissionMode.ASCENT,
            MissionMode.DESCENT,
        ):
            return

        target_ascent_m = self.config.global_.initial_height
        target_descent_m = self.config.global_.final_height
        x_right = x_left + bar_width
        y_bottom = y_top + bar_height
        total_range = target_ascent_m - target_descent_m
        progress = np.clip((pos_z - target_descent_m) / total_range, 0.0, 1.0)
        fill_color = current_mode_color
        fill_h = int(bar_height * progress)
        fill = y_bottom - fill_h
        # Background (dark)
        cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), bg_color, cv2.FILLED)

        # Normalize progress
        if current_mode == MissionMode.ASCENT:

            cv2.rectangle(
                frame, (x_left, fill), (x_right, y_bottom), fill_color, cv2.FILLED
            )
        else:  # DESCENT
            cv2.rectangle(
                frame, (x_left, fill), (x_right, y_bottom), fill_color, cv2.FILLED
            )

        # Border
        cv2.rectangle(
            frame, (x_left, y_top), (x_right, y_bottom), border_color, thickness=2
        )

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        thick = 1
        alt_text = f"{pos_z:.1f} cm"
        alt_size, _ = cv2.getTextSize(alt_text, font, font_scale, thick)
        alt_x = x_left + (bar_width - alt_size[0]) // 2
        alt_y = y_bottom + 20
        cv2.putText(
            frame,
            alt_text,
            (alt_x, alt_y),
            font,
            font_scale,
            (50, 50, 50),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            alt_text,
            (alt_x, alt_y),
            font,
            font_scale,
            current_mode_color,
            thick,
            cv2.LINE_AA,
        )

    def draw_mode_banner(self, frame: np.ndarray, current_mode: MissionMode) -> None:
        """
        Draws a centered top banner with black background + colored text.
        """
        text = self.mode_display_name(current_mode)
        mode_key = current_mode.value
        bg_color = (30, 30, 30)
        accent_color = self.mode_colors[mode_key]

        # Font settings
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 2

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Banner padding
        padding_x = 10
        padding_y = 6
        banner_w = text_w + 2 * padding_x
        banner_h = text_h + baseline + 2 * padding_y

        # Center at top
        frame_h, frame_w = frame.shape[:2]
        banner_x = (frame_w - banner_w) // 2 + 10
        banner_y = 12  # distance from top

        # Draw background rectangle (black/very dark)
        cv2.rectangle(
            frame,
            (banner_x, banner_y),
            (banner_x + banner_w, banner_y + banner_h),
            bg_color,
            cv2.FILLED,
        )

        cv2.rectangle(
            frame,
            (banner_x, banner_y),
            (banner_x + banner_w, banner_y + banner_h),
            accent_color,
            thickness=2,
        )

        # Draw text (centered inside banner)
        text_x = banner_x + padding_x
        text_y = banner_y + padding_y + text_h

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            accent_color,
            thickness,
            cv2.LINE_AA,
        )

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
        current_mode_color = self.mode_colors[current_mode.value]

        img = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        center = np.array([cx, cy])

        pos_x = state.position[0]
        pos_y = state.position[1]
        pos_z = state.position[2]

        feature_img = frame.copy()
        # Draw points
        for pt in new_pts:
            x, y = map(int, pt.ravel())
            cv2.circle(feature_img, (x, y), 4, (100, 255, 200), -1)
        # Draw center indicator
        size = 20
        cv2.line(feature_img, (cx - size, cy), (cx + size, cy), (300, 300, 300), 3)
        cv2.line(feature_img, (cx, cy - size), (cx, cy + size), (300, 300, 300), 3)
        img = cv2.addWeighted(img, 0.5, feature_img, 0.5, 0)

        # Draw new trails
        for new, old in zip(new_pts, old_pts):
            a, b = map(int, new.ravel())
            c, d = map(int, old.ravel())
            cv2.line(trail_mask, (a, b), (c, d), (100, 255, 200), 2)

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
        if not mission_manager.in_landing_phase:

            if current_mode == MissionMode.NAVIGATION:
                direction = np.array([dx_to_search_zone_cm, -dy_to_search_zone_cm])
                direction /= dist_to_search_zone_cm
                max_len = self.config.navigation.arrow_max_length_px
                arrow_length_px = min(
                    cm_to_pixels(dist_to_search_zone_cm, pos_z, self.focal_px), max_len
                )
                end_point = center + (direction * arrow_length_px).astype(int)
                cv2.arrowedLine(
                    img,
                    tuple(center.astype(int)),
                    tuple(map(int, end_point)),
                    current_mode_color,
                    6,
                    tipLength=0.3,
                )
                cv2.putText(
                    img,
                    f"Target: {dist_to_search_zone_cm:.1f} cm",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    current_mode_color,
                    2,
                )

        # === Heatmap + Search Zone (only in landing phases) ===
        else:
            # Re-compute original target projection here (safe, inside this block)
            orig_target_x_cm = self.config.navigation.planned_route_dx
            orig_target_y_cm = self.config.navigation.planned_route_dy
            orig_remaining_x = orig_target_x_cm - pos_x
            orig_remaining_y = orig_target_y_cm - pos_y

            target_px_x = w // 2 + int(
                cm_to_pixels(orig_remaining_x, pos_z, self.focal_px)
            )
            target_px_y = h // 2 - int(
                cm_to_pixels(orig_remaining_y, pos_z, self.focal_px)
            )
            target_px = (target_px_x, target_px_y)

            outer_radius_px = int(
                cm_to_pixels(
                    self.config.navigation.search_zone_outer_thresh,
                    pos_z,
                    self.focal_px,
                )
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

            img = overlay

            # Search radius circle (centered on original target)
            cv2.circle(img, target_px, outer_radius_px, current_mode_color, 4)

            if current_mode == MissionMode.LANDING_APPROACH:
                landing_site_pixel = center + np.array(
                    [
                        cm_to_pixels(
                            dx_to_locked_landing_target_cm, pos_z, self.focal_px
                        ),
                        -cm_to_pixels(
                            dy_to_locked_landing_target_cm, pos_z, self.focal_px
                        ),
                    ]
                ).astype("int")
                cv2.circle(
                    img,
                    landing_site_pixel,
                    int(
                        cm_to_pixels(
                            self.config.navigation.pos_tolerance_cm,
                            pos_z,
                            self.focal_px,
                        )
                    ),
                    current_mode_color,
                    4,
                )
                dist_to_locked_landing_target_cm = np.hypot(
                    dx_to_locked_landing_target_cm, dy_to_locked_landing_target_cm
                )

                cv2.putText(
                    img,
                    f"Safe Site: {dist_to_locked_landing_target_cm:.1f} cm",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    current_mode_color,
                    2,
                )

            elif current_mode == MissionMode.DESCENT:
                landing_site_pixel = center + np.array(
                    [
                        cm_to_pixels(
                            dx_to_locked_landing_target_cm, pos_z, self.focal_px
                        ),
                        -cm_to_pixels(
                            dy_to_locked_landing_target_cm, pos_z, self.focal_px
                        ),
                    ]
                ).astype("int")
                cv2.circle(
                    img,
                    landing_site_pixel,
                    int(
                        cm_to_pixels(
                            self.config.navigation.pos_tolerance_cm,
                            pos_z,
                            self.focal_px,
                        )
                    ),
                    current_mode_color,
                    4,
                )

            elif current_mode == MissionMode.HOVERING:
                landing_site_pixel = center + np.array(
                    [
                        cm_to_pixels(
                            dx_to_locked_landing_target_cm, pos_z, self.focal_px
                        ),
                        -cm_to_pixels(
                            dy_to_locked_landing_target_cm, pos_z, self.focal_px
                        ),
                    ]
                ).astype("int")
                # Show countdown instead of arrow
                hover_remaining = mission_manager.get_hover_remaining_s()
                cv2.putText(
                    img,
                    f"{hover_remaining:.1f}",
                    landing_site_pixel,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (50, 50, 50),
                    8,
                )
                cv2.putText(
                    img,
                    f"{hover_remaining:.1f}",
                    landing_site_pixel,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    current_mode_color,
                    2,
                )
                cv2.circle(
                    img,
                    landing_site_pixel,
                    int(
                        cm_to_pixels(
                            self.config.navigation.pos_tolerance_cm,
                            pos_z,
                            self.focal_px,
                        )
                    ),
                    current_mode_color,
                    4,
                )
            elif current_mode == MissionMode.LANDED_SAFE:
                cv2.putText(
                    img,
                    f"MISSION COMPLETE",
                    (100, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    15,
                )
                cv2.putText(
                    img,
                    f"MISSION COMPLETE",
                    (100, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    current_mode_color,
                    4,
                )

        # Standard info
        cv2.putText(
            img,
            f"Pos: ({pos_x:+.1f}, {pos_y:+.1f}, {pos_z:.1f}) cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
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
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            2,
        )

        self.draw_mode_banner(img, current_mode)

        self.draw_altitude_progress_bars(img, current_mode, current_mode_color, pos_z)

        return img
