import numpy as np


class Commander:
    @staticmethod
    def issue_commands(
        flow_dx_cm: float,
        flow_dy_cm: float,
        safe_dx_cm: float | None,
        safe_dy_cm: float | None,
        remaining_x_cm: float,
        remaining_y_cm: float,
        in_landing_mode: bool,
    ):
        # Always show current ego-motion from optical flow
        mag = abs(flow_dx_cm) + abs(flow_dy_cm)
        if mag > 0.5:
            dist = mag * 15
            dir_x = "right" if flow_dx_cm > 0 else "left"
            dir_y = "forward" if flow_dy_cm > 0 else "backward"
            direction = dir_x if abs(flow_dx_cm) > abs(flow_dy_cm) else dir_y
            print(f"→ Current motion: Move {direction} {dist:.0f} cm")

        remaining_dist = np.hypot(remaining_x_cm, remaining_y_cm)

        if remaining_dist < 0.5:
            print("→ Arrived at target area")
            return

        if not in_landing_mode:
            # Navigation phase: direct toward target
            if abs(remaining_x_cm) > abs(remaining_y_cm):
                main_dir = "right" if remaining_x_cm > 0 else "left"
                dist = abs(remaining_x_cm)
            else:
                main_dir = "forward" if remaining_y_cm > 0 else "backward"
                dist = abs(remaining_y_cm)
            print(
                f"→ Navigate to target: Move {main_dir} {dist:.0f} cm ({remaining_dist:.0f} cm total)"
            )
        else:
            # Landing phase
            if safe_dx_cm is not None and safe_dy_cm is not None:
                if abs(safe_dx_cm) > 5 or abs(safe_dy_cm) > 5:
                    dir_x = "right" if safe_dx_cm > 0 else "left"
                    dir_y = "forward" if safe_dy_cm > 0 else "backward"
                    main_dir = dir_x if abs(safe_dx_cm) > abs(safe_dy_cm) else dir_y
                    dist = max(abs(safe_dx_cm), abs(safe_dy_cm))
                    print(f"→ Land near target: Move {main_dir} {dist:.0f} cm")
                else:
                    print("→ At target: Current position is safe for landing")
            else:
                print("→ NO SAFE LANDING ZONE NEAR TARGET")
