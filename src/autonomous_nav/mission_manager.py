import numpy as np
import time
from enum import Enum
from autonomous_nav.config import AppConfig


class MissionMode(Enum):
    NAVIGATION = "navigation"
    LANDING_APPROACH = "landing_approach"
    LANDED_SAFE = "landed_safe"
    NO_SAFE_ZONE = "no_safe_zone"


class MissionManager:
    def __init__(self, config: AppConfig):
        self.config = config
        self.current_mode = MissionMode.NAVIGATION
        self.previous_mode = None

        self.target_x_cm = config.navigation.target_offset_x_cm
        self.target_y_cm = config.navigation.target_offset_y_cm
        self.arrival_inner_threshold_cm = config.navigation.arrival_inner_threshold_cm
        self.arrival_outer_threshold_cm = config.navigation.arrival_outer_threshold_cm
        self.landed_threshold_cm = (
            15.0  # Distance to declare "very close" for landing confirmation
        )

        # New: Configurable stability (hysteresis) in frames
        self.stability_frames = config.navigation.landing_mode_stability_frames

        # Landing site lock (world coordinates)
        self.landing_target_x_cm: float | None = None
        self.landing_target_y_cm: float | None = None

        # Hover confirmation timer
        self.hover_start_time: float | None = None

        # Hysteresis counters
        self.safe_count = 0  # Consecutive frames with safe_center_px available
        self.no_safe_count = 0  # Consecutive frames with no safe_center_px

        # Tolerances from config
        self.pos_tolerance_cm = config.navigation.pos_tolerance_cm
        self.vel_tolerance_cm_s = config.navigation.vel_tolerance_cm_s
        self.hover_duration_s = config.navigation.hover_duration_s

    def reset(self):
        self.current_mode = MissionMode.NAVIGATION
        self.previous_mode = None
        self._reset_landing()

    def update(
        self,
        pos_x: float,
        pos_y: float,
        vel_x: float,
        vel_y: float,
        safe_dx_cm: float | None,
        safe_dy_cm: float | None,
        safe_center_px: tuple[int, int] | None,
    ) -> MissionMode:
        remaining_dist_to_target = np.hypot(
            self.target_x_cm - pos_x, self.target_y_cm - pos_y
        )
        inside_inner = remaining_dist_to_target < self.arrival_inner_threshold_cm
        outside_outer = remaining_dist_to_target > self.arrival_outer_threshold_cm
        very_close = remaining_dist_to_target < self.landed_threshold_cm

        has_safe_spot = safe_center_px is not None
        vel_mag = np.hypot(vel_x, vel_y)

        new_mode = self.current_mode

        if self.current_mode == MissionMode.NAVIGATION:
            if inside_inner:
                if has_safe_spot:
                    self.safe_count += 1
                    self.no_safe_count = 0
                    if self.safe_count >= self.stability_frames:
                        # Commit to landing site
                        self.landing_target_x_cm = pos_x + (safe_dx_cm or 0.0)
                        self.landing_target_y_cm = pos_y + (safe_dy_cm or 0.0)
                        new_mode = MissionMode.LANDING_APPROACH
                        self.safe_count = 0
                else:
                    self.no_safe_count += 1
                    self.safe_count = 0
                    if self.no_safe_count >= self.stability_frames:
                        new_mode = MissionMode.NO_SAFE_ZONE
                        self.no_safe_count = 0
            else:
                # Far from target — reset counters
                self._reset_counters()

        elif self.current_mode == MissionMode.LANDING_APPROACH:
            if outside_outer:  # NEW: Use outer threshold to exit
                self._reset_landing()
                new_mode = MissionMode.NAVIGATION
            else:
                if has_safe_spot and very_close:
                    # Check hover stability for landing confirmation
                    dist_to_landing = np.hypot(
                        self.landing_target_x_cm - pos_x,
                        self.landing_target_y_cm - pos_y,
                    )
                    is_hovering = (
                        dist_to_landing < self.pos_tolerance_cm
                        and vel_mag < self.vel_tolerance_cm_s
                    )

                    if is_hovering:
                        if self.hover_start_time is None:
                            self.hover_start_time = time.time()
                        elif (
                            time.time() - self.hover_start_time >= self.hover_duration_s
                        ):
                            new_mode = MissionMode.LANDED_SAFE
                    else:
                        self.hover_start_time = None

                if not has_safe_spot:
                    self.no_safe_count += 1
                    self.safe_count = 0
                    if self.no_safe_count >= self.stability_frames:
                        new_mode = MissionMode.NO_SAFE_ZONE
                        self.no_safe_count = 0
                else:
                    self.safe_count += 1
                    self.no_safe_count = 0
                    # Optionally re-lock to better spot if significantly improved
                    # (not implemented here to keep lock stable)

        elif self.current_mode == MissionMode.NO_SAFE_ZONE:
            if outside_outer:  # NEW: Use outer threshold to exit
                self._reset_landing()
                new_mode = MissionMode.NAVIGATION
            elif has_safe_spot:
                self.safe_count += 1
                self.no_safe_count = 0
                if self.safe_count >= self.stability_frames:
                    # Recover and lock new safe site
                    self.landing_target_x_cm = pos_x + (safe_dx_cm or 0.0)
                    self.landing_target_y_cm = pos_y + (safe_dy_cm or 0.0)
                    new_mode = MissionMode.LANDING_APPROACH
                    self.safe_count = 0
            else:
                self.safe_count = 0
                # Stay in NO_SAFE_ZONE

        elif self.current_mode == MissionMode.LANDED_SAFE:
            if outside_outer:  # NEW: Use outer threshold to exit
                self._reset_landing()
                new_mode = MissionMode.NAVIGATION

        # Mode change logging
        if new_mode != self.current_mode:
            self.previous_mode = self.current_mode
            self.current_mode = new_mode
            print(
                f"[MissionManager] Mode change: {self.previous_mode.value} → {new_mode.value}"
            )

        return self.current_mode

    def _reset_landing(self):
        self.landing_target_x_cm = None
        self.landing_target_y_cm = None
        self.hover_start_time = None
        self._reset_counters()

    def _reset_counters(self):
        self.safe_count = 0
        self.no_safe_count = 0

    @property
    def in_landing_phase(self) -> bool:
        return self.current_mode in (
            MissionMode.LANDING_APPROACH,
            MissionMode.LANDED_SAFE,
            MissionMode.NO_SAFE_ZONE,
        )
