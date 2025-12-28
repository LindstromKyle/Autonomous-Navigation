class Commander:
    @staticmethod
    def issue_commands(
        flow_dx_cm: float,
        flow_dy_cm: float,
        safe_dx_cm: float | None,
        safe_dy_cm: float | None,
    ):
        mag = abs(flow_dx_cm) + abs(flow_dy_cm)
        if mag > 0.5:
            dist = mag * 15
            dir_x = "right" if flow_dx_cm > 0 else "left"
            dir_y = "forward" if flow_dy_cm > 0 else "backward"
            direction = dir_x if abs(flow_dx_cm) > abs(flow_dy_cm) else dir_y
            print(f"→ Move {direction} {dist:.0f} cm")

        if safe_dx_cm is not None and safe_dy_cm is not None:
            if abs(safe_dx_cm) > 5 or abs(safe_dy_cm) > 5:
                dir_x = "right" if safe_dx_cm > 0 else "left"
                dir_y = "forward" if safe_dy_cm > 0 else "backward"
                main_dir = dir_x if abs(safe_dx_cm) > abs(safe_dy_cm) else dir_y
                dist = max(abs(safe_dx_cm), abs(safe_dy_cm))
                print(f"→ Safe landing zone: Move {main_dir} {dist:.0f} cm")
            else:
                print("→ Current position is safe for landing")
        else:
            print("No safe landing zone detected")
