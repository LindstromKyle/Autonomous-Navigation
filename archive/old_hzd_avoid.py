class DensityBasedHazardAvoidance(HazardAvoidance):
    def __init__(self, config: HazardConfig, global_config: AppConfig):
        self.config = config
        self.pixels_per_cm = global_config.global_.pixels_per_cm

    def detect(
        self,
        features: np.ndarray,
        frame_height: int,
        frame_width: int,
        target_px: tuple[int, int] | None = None,
    ) -> tuple[float | None, float | None, np.ndarray, tuple[int, int] | None]:
        grid_size = self.config.grid_size
        cell_h = frame_height // grid_size
        cell_w = frame_width // grid_size
        density = np.zeros((grid_size, grid_size), dtype=int)

        if len(features) == 0:
            hazard_mask = density > self.config.threshold
            return None, None, hazard_mask, None

        for x, y in features:
            gx = min(int(x // cell_w), grid_size - 1)
            gy = min(int(y // cell_h), grid_size - 1)
            density[gy, gx] += 1

        hazard_mask = density > self.config.threshold

        if self.config.exclude_boundaries and grid_size > 2:
            hazard_mask[0, :] = True
            hazard_mask[-1, :] = True
            hazard_mask[:, 0] = True
            hazard_mask[:, -1] = True

        safe_mask = ~hazard_mask
        if not np.any(safe_mask):
            return None, None, hazard_mask, None

        labeled, num = label(safe_mask)
        if num == 0:
            return None, None, hazard_mask, None

        cell_centers = []
        sizes = []
        for i in range(1, num + 1):
            safe_slice = find_objects(labeled == i)[0]
            cy = (safe_slice[0].start + safe_slice[0].stop - 1) // 2
            cx = (safe_slice[1].start + safe_slice[1].stop - 1) // 2
            center_px = (cx * cell_w + cell_w // 2, cy * cell_h + cell_h // 2)
            size = np.sum(labeled == i)
            cell_centers.append(center_px)
            sizes.append(size)

        if target_px is None:
            # Standard mode: largest safe zone (always return one if any exist)
            if num > 0:
                best_idx = np.argmax(sizes)
                safe_center_px = cell_centers[best_idx]
            else:
                safe_center_px = None
        else:
            # Landing mode: only accept zones close to target_px
            distances = [
                np.hypot(cx - target_px[0], cy - target_px[1])
                for cx, cy in cell_centers
            ]
            proximity_threshold_px = min(frame_width, frame_height) // 2

            candidates = [
                i for i, d in enumerate(distances) if d < proximity_threshold_px
            ]

            if candidates:
                # Pick largest among proximate ones
                best_idx = candidates[np.argmax([sizes[i] for i in candidates])]
                safe_center_px = cell_centers[best_idx]
            else:
                # No safe zone near target → explicitly return None
                safe_center_px = None

        if safe_center_px is None:
            dx_cm = dy_cm = None
        else:
            dx_px = safe_center_px[0] - frame_width // 2
            dy_px = safe_center_px[1] - frame_height // 2
            dx_cm = pixels_to_cm(dx_px, self.pixels_per_cm)
            dy_cm = pixels_to_cm(dy_px, self.pixels_per_cm)

        return dx_cm, dy_cm, hazard_mask, safe_center_px
