import numpy as np
import time
from enum import Enum
from autonomous_nav.config import AppConfig


class MissionMode(Enum):
    ASCENT = "ascent"
    NAVIGATION = "navigation"
    SEARCHING = "searching"
    LANDING_APPROACH = "landing_approach"
    DESCENT = "descent"
    HOVERING = "hovering"
    LANDED_SAFE = "landed_safe"
    NO_SAFE_ZONE = "no_safe_zone"


class MissionManager:
    """
    Mission manager module
    """

    def __init__(self, config: AppConfig):
        # Set up params from config
        self.config = config
        self.current_mode = MissionMode.ASCENT
        self.previous_mode = None
        self.planned_route_dx = config.navigation.planned_route_dx
        self.planned_route_dy = config.navigation.planned_route_dy
        self.search_zone_inner_thresh = config.navigation.search_zone_inner_thresh
        self.search_zone_outer_thresh = config.navigation.search_zone_outer_thresh
        self.stability_frames = config.navigation.landing_mode_stability_frames
        self.pos_tolerance_cm = config.navigation.pos_tolerance_cm
        self.hover_duration_s = config.navigation.hover_duration_s

        # Landing site lock
        self.locked_landing_target_x_cm: float | None = None
        self.locked_landing_target_y_cm: float | None = None

        # Hover confirmation timer
        self.hover_start_time: float | None = None

        # Hysteresis counters
        self.safe_count = 0
        self.no_safe_count = 0

        # Landing site coords
        self.safe_xs = []
        self.safe_ys = []

    def reset(self):
        """
        Reset mission manager back to NAVIGATION mode
        """
        self.current_mode = MissionMode.NAVIGATION
        self.previous_mode = None
        self._reset_landing()

    def update(
        self,
        pos_x: float,
        pos_y: float,
        pos_z: float,
        current_frame_safe_dx_cm: float | None,
        current_frame_safe_dy_cm: float | None,
        current_frame_safe_px: tuple[int, int] | None,
    ) -> MissionMode:
        """
        Check current state and updates mission manager if criteria for new mode
        are met
        """

        dist_to_search_center = np.hypot(
            self.planned_route_dx - pos_x, self.planned_route_dy - pos_y
        )
        inside_inner = dist_to_search_center < self.search_zone_inner_thresh
        outside_outer = dist_to_search_center > self.search_zone_outer_thresh
        has_safe_spot = current_frame_safe_px is not None
        new_mode = self.current_mode

        # Check ascent criteria
        if self.current_mode == MissionMode.ASCENT:
            if pos_z >= self.config.global_.initial_height:
                new_mode = MissionMode.NAVIGATION

        # Check navigation criteria
        elif self.current_mode == MissionMode.NAVIGATION:
            if inside_inner:
                new_mode = MissionMode.SEARCHING
            else:
                # Far from target, reset counters
                self._reset_counters()

        # Check searching criteria
        elif self.current_mode == MissionMode.SEARCHING:
            if has_safe_spot:
                self.safe_count += 1
                self.no_safe_count = 0
                self.safe_xs.append(pos_x + current_frame_safe_dx_cm)
                self.safe_ys.append(pos_y + current_frame_safe_dy_cm)

                if self.safe_count >= self.stability_frames:
                    # Commit to landing site
                    self.locked_landing_target_x_cm = np.median(self.safe_xs)
                    self.locked_landing_target_y_cm = np.median(self.safe_ys)
                    new_mode = MissionMode.LANDING_APPROACH
                    self.safe_count = 0
                    self.safe_xs = []
                    self.safe_ys = []
            else:
                self.no_safe_count += 1
                self.safe_count = 0
                if self.no_safe_count >= self.stability_frames:
                    new_mode = MissionMode.NO_SAFE_ZONE
                    self.no_safe_count = 0

        # Check landing approach criteria
        elif self.current_mode == MissionMode.LANDING_APPROACH:
            if outside_outer:
                self._reset_landing()
                new_mode = MissionMode.NAVIGATION
            else:
                # Check distance to locked landing target
                dist_to_landing = np.hypot(
                    self.locked_landing_target_x_cm - pos_x,
                    self.locked_landing_target_y_cm - pos_y,
                )
                if dist_to_landing < self.pos_tolerance_cm:
                    new_mode = MissionMode.DESCENT

                if not has_safe_spot:
                    self.no_safe_count += 1
                    self.safe_count = 0
                    if self.no_safe_count >= self.stability_frames:
                        new_mode = MissionMode.NO_SAFE_ZONE
                        self.no_safe_count = 0
                else:
                    self.safe_count += 1
                    self.no_safe_count = 0

        # Check descent criteria
        elif self.current_mode == MissionMode.DESCENT:
            if pos_z <= self.config.global_.final_height:
                new_mode = MissionMode.HOVERING
                if self.hover_start_time is None:
                    self.hover_start_time = time.time()

        # Check hovering criteria
        elif self.current_mode == MissionMode.HOVERING:
            # Check distance to locked landing target
            dist_to_landing = np.hypot(
                self.locked_landing_target_x_cm - pos_x,
                self.locked_landing_target_y_cm - pos_y,
            )
            if dist_to_landing >= self.pos_tolerance_cm:
                new_mode = MissionMode.LANDING_APPROACH
                self.hover_start_time = None
            elif pos_z > self.config.global_.final_height + 1.5:
                new_mode = MissionMode.DESCENT
            else:
                elapsed = time.time() - self.hover_start_time
                if elapsed >= self.hover_duration_s:
                    new_mode = MissionMode.LANDED_SAFE

            if not has_safe_spot:
                self.no_safe_count += 1
                self.safe_count = 0
                if self.no_safe_count >= self.stability_frames:
                    new_mode = MissionMode.NO_SAFE_ZONE
                    self.no_safe_count = 0
            else:
                self.safe_count += 1
                self.no_safe_count = 0

        # Check no safe zone criteria
        elif self.current_mode == MissionMode.NO_SAFE_ZONE:
            if outside_outer:
                self._reset_landing()
                new_mode = MissionMode.NAVIGATION
            elif has_safe_spot:
                self.safe_count += 1
                self.no_safe_count = 0
                new_mode = MissionMode.SEARCHING
            else:
                self.safe_count = 0

        # Check landed safe criteria
        elif self.current_mode == MissionMode.LANDED_SAFE:
            if outside_outer:
                self._reset_landing()
                new_mode = MissionMode.NAVIGATION

        # Mode change logging
        if new_mode != self.current_mode:
            self.previous_mode = self.current_mode
            self.current_mode = new_mode
            print(
                f"[MissionManager] Mode change: {self.previous_mode.value} to {new_mode.value}"
            )

    def _reset_landing(self):
        """
        Undo the lock on the target landing site, reset stability frame counters
        """
        self.locked_landing_target_x_cm = None
        self.locked_landing_target_y_cm = None
        self.hover_start_time = None
        self._reset_counters()

    def _reset_counters(self):
        """
        Set stability frame counters to zero
        """
        self.safe_count = 0
        self.no_safe_count = 0

    @property
    def in_landing_phase(self) -> bool:
        """
        Helper function to check if drone is in any of the landing modes
        """
        return self.current_mode in (
            MissionMode.LANDING_APPROACH,
            MissionMode.SEARCHING,
            MissionMode.DESCENT,
            MissionMode.HOVERING,
            MissionMode.LANDED_SAFE,
            MissionMode.NO_SAFE_ZONE,
        )

    def is_hovering(self) -> bool:
        """
        Helper function to determin if drone is currently hovering
        """
        return self.hover_start_time is not None

    def get_hover_remaining_s(self) -> float | None:
        """
        Computes how long is left in the hover countdown
        """
        if self.hover_start_time is None:
            return None
        elapsed = time.time() - self.hover_start_time
        return max(0.0, self.hover_duration_s - elapsed)
