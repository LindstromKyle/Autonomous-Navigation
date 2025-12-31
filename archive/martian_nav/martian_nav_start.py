import cv2
import numpy as np
from picamera2 import Picamera2
import time
from scipy.ndimage import label, find_objects

# GENERAL PARAMS
PIXELS_PER_CM = 14.0
MIN_FEATURES = 40
NUM_FRAMES_REDETECT = 20
FRAME_SIZE = (640, 480)

# FEATURE DETECTION PARAMS
MAX_CORNERS = 200
CORNER_QUALITY_LEVEL = 0.1
MIN_CORNER_DISTANCE = 7
CORNER_BLOCK_SIZE = 7

# OPTICAL FLOW PARAMS
WINDOW_SIZE = (15, 15)
MAX_LEVEL = 2

# HAZARD AVOIDANCE PARAMS
HAZARD_GRID_SIZE = 8
HAZARD_THRESHOLD = 2
EXCLUDE_BOUNDARIES = True

# CLAHE PARAMS
CLIP_LIMIT_NORMALIZED = 0.01
CLAHE_TILE_GRID_SIZE = (8, 8)

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": FRAME_SIZE, "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()
time.sleep(2)


old_frame = picam2.capture_array()
old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
old_features = cv2.goodFeaturesToTrack(
    old_frame_gray,
    mask=None,
    maxCorners=MAX_CORNERS,
    qualityLevel=CORNER_QUALITY_LEVEL,
    minDistance=MIN_CORNER_DISTANCE,
    blockSize=CORNER_BLOCK_SIZE,
)
mask = np.zeros_like(old_frame)

# Preprocessing
nbins = 256
clip_limit = CLIP_LIMIT_NORMALIZED * nbins
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=CLAHE_TILE_GRID_SIZE)

sensor_pos_x = sensor_pos_y = 0.0
frame_count = 0

print("\n=== Martian Rover Navigation ===")
print("Stable feature refresh and hazard detection enabled.\n")

while True:
    new_frame = picam2.capture_array()
    new_frame_gray_raw = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)

    # Pre-processing
    new_frame_gray = clahe.apply(new_frame_gray_raw)
    # new_frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), 0)  # Optional

    frame_count += 1

    flow_dx = flow_dy = 0.0
    valid_new_features = np.empty((0, 1, 2))

    if old_features is not None and len(old_features) > 0:
        new_features, valid_flag, _ = cv2.calcOpticalFlowPyrLK(
            old_frame_gray,
            new_frame_gray,
            old_features,
            None,
            winSize=WINDOW_SIZE,
            maxLevel=MAX_LEVEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        if new_features is not None:
            valid_new_features = new_features[valid_flag == 1]
            valid_old_features = old_features[valid_flag == 1]

            if len(valid_new_features) > 5:
                flow = valid_new_features.reshape(-1, 2) - valid_old_features.reshape(
                    -1, 2
                )
                flow_dx = np.median(flow[:, 0])
                flow_dy = np.median(flow[:, 1])

    # Default: continue with tracked points
    old_features = (
        valid_new_features.reshape(-1, 1, 2) if valid_new_features.size > 0 else None
    )

    # Refresh: replace with fresh strong features if low or periodic
    if (
        old_features is None or len(old_features) < MIN_FEATURES
    ) or frame_count % NUM_FRAMES_REDETECT == 0:
        redetect_features = cv2.goodFeaturesToTrack(
            new_frame_gray,
            mask=None,
            maxCorners=MAX_CORNERS,
            qualityLevel=CORNER_QUALITY_LEVEL,
            minDistance=MIN_CORNER_DISTANCE,
            blockSize=CORNER_BLOCK_SIZE,
        )
        if redetect_features is not None:
            old_features = redetect_features
            mask = np.zeros_like(old_frame)  # Clear trails on full refresh

    # Update position
    flow_dx_cm = -flow_dx / PIXELS_PER_CM
    flow_dy_cm = flow_dy / PIXELS_PER_CM
    sensor_pos_x += flow_dx_cm
    sensor_pos_y += flow_dy_cm

    # --- Hazard Detection (High density = rocks/hazards; sparse = safe) ---
    frame_height, frame_width = new_frame_gray.shape
    hazard_cell_h = frame_height // HAZARD_GRID_SIZE
    hazard_cell_w = frame_width // HAZARD_GRID_SIZE
    hazard_density_grid = np.zeros((HAZARD_GRID_SIZE, HAZARD_GRID_SIZE), dtype=int)

    if len(valid_new_features) > 0:
        hazards = valid_new_features.reshape(-1, 2)
        for x, y in hazards:
            hazard_grid_x = min(int(x // hazard_cell_w), HAZARD_GRID_SIZE - 1)
            hazard_grid_y = min(int(y // hazard_cell_h), HAZARD_GRID_SIZE - 1)
            hazard_density_grid[hazard_grid_y, hazard_grid_x] += 1

    hazard_mask = hazard_density_grid > HAZARD_THRESHOLD

    # Exclude boundary cells from safe mask (force inner safe zones)
    if EXCLUDE_BOUNDARIES and HAZARD_GRID_SIZE > 2:
        hazard_mask[0, :] = True  # Top row
        hazard_mask[-1, :] = True  # Bottom row
        hazard_mask[:, 0] = True  # Left column
        hazard_mask[:, -1] = True  # Right column

    safe_mask = ~hazard_mask

    # Find largest safe zone using connected components
    if np.any(safe_mask):
        labeled_safe_zones, num_safe = label(safe_mask)
        if num_safe > 0:
            safe_zone_sizes = [
                np.sum(labeled_safe_zones == i) for i in range(1, num_safe + 1)
            ]
            largest_safe_zone_id = np.argmax(safe_zone_sizes) + 1

            # Get bounding box of largest safe blob
            safe_slice = find_objects(labeled_safe_zones == largest_safe_zone_id)[0]

            # Geometric center (in grid coordinates)
            geo_center_y = (safe_slice[0].start + safe_slice[0].stop - 1) // 2
            geo_center_x = (safe_slice[1].start + safe_slice[1].stop - 1) // 2

            # Pixel position of geometric center
            geo_px = (
                geo_center_x * hazard_cell_w + hazard_cell_w // 2,
                geo_center_y * hazard_cell_h + hazard_cell_h // 2,
            )

            # Now find the closest safe (non-hazard) grid cell to this center
            # Only consider cells that are actually in the safe blob
            safe_y, safe_x = np.where(labeled_safe_zones == largest_safe_zone_id)
            safe_coords = list(zip(safe_y, safe_x))

            if not safe_coords:
                safe_center_px = None
                dx_to_safe_zone_cm = dy_to_safe_zone_cm = None
            else:
                # Find distances from geo center to each safe cell center
                cell_centers = [
                    (
                        sx * hazard_cell_w + hazard_cell_w // 2,
                        sy * hazard_cell_h + hazard_cell_h // 2,
                    )
                    for sy, sx in safe_coords
                ]
                distances = [
                    np.hypot(cx - geo_px[0], cy - geo_px[1]) for cx, cy in cell_centers
                ]
                closest_idx = np.argmin(distances)
                safe_center_px = cell_centers[closest_idx]

                # Relative position in cm for commands
                dx_to_safe_zone_px = safe_center_px[0] - frame_width // 2
                dy_to_safe_zone_px = safe_center_px[1] - frame_height // 2
                dx_to_safe_zone_cm = dx_to_safe_zone_px / PIXELS_PER_CM
                dy_to_safe_zone_cm = dy_to_safe_zone_px / PIXELS_PER_CM
        else:
            safe_center_px = None
            dx_to_safe_zone_cm = dy_to_safe_zone_cm = None
    else:
        safe_center_px = None
        dx_to_safe_zone_cm = dy_to_safe_zone_cm = None

    # Draw trails and points
    for valid_new, valid_old in zip(valid_new_features, valid_old_features):
        a, b = valid_new.ravel().astype(int)
        c, d = valid_old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        new_frame = cv2.circle(new_frame, (a, b), 5, (0, 0, 255), -1)

    cv2_img = cv2.add(new_frame, mask)

    # Overlay hazards (red) and safe zone (green circle)
    for grid_row in range(HAZARD_GRID_SIZE):
        for grid_col in range(HAZARD_GRID_SIZE):
            if hazard_mask[grid_row, grid_col] == True:
                top_left = (grid_col * hazard_cell_w, grid_row * hazard_cell_h)
                bottom_right = (
                    (grid_col + 1) * hazard_cell_w,
                    (grid_row + 1) * hazard_cell_h,
                )
                cv2.rectangle(cv2_img, top_left, bottom_right, (0, 0, 255), 2)

    if safe_center_px is not None:
        cv2.circle(cv2_img, safe_center_px, 30, (0, 255, 0), 3)  # Larger green circle

    # Display info
    cv2.putText(
        cv2_img,
        f"Pos: ({sensor_pos_x:+.1f}, {sensor_pos_y:+.1f}) cm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        cv2_img,
        f"Features: {len(valid_new_features)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        cv2_img,
        f"Hazards: {np.sum(hazard_mask)}/{HAZARD_GRID_SIZE**2}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        cv2_img,
        "r = reset pos | q = quit",
        (10, cv2_img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    cv2.imshow("Martian Rover Navigation", cv2_img)

    # Console commands toward safe zone
    if frame_count % 30 == 0:
        mag = abs(flow_dx_cm) + abs(flow_dy_cm)
        if mag > 0.5:
            dist = mag * 15
            dir_x = "right" if flow_dx_cm > 0 else "left"
            dir_y = "forward" if flow_dy_cm > 0 else "backward"
            direction = dir_x if abs(flow_dx_cm) > abs(flow_dy_cm) else dir_y
            print(f"→ Move {direction} {dist:.0f} cm")

        if dx_to_safe_zone_cm is not None and dy_to_safe_zone_cm is not None:
            # Safe zone command
            if abs(dx_to_safe_zone_cm) > 5 or abs(dy_to_safe_zone_cm) > 5:
                dir_x = "right" if dx_to_safe_zone_cm > 0 else "left"
                dir_y = "forward" if dy_to_safe_zone_cm > 0 else "backward"
                main_dir = (
                    dir_x
                    if abs(dx_to_safe_zone_cm) > abs(dy_to_safe_zone_cm)
                    else dir_y
                )
                dist = max(abs(dx_to_safe_zone_cm), abs(dy_to_safe_zone_cm))
                print(f"→ Safe landing zone: Move {main_dir} {dist:.0f} cm")
            else:
                print("→ Current position is safe for landing")
                pass
        else:
            print("No safe landing zone detected")

    old_frame_gray = new_frame_gray.copy()

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        sensor_pos_x = sensor_pos_y = 0.0
        print("Position reset")

picam2.stop()
cv2.destroyAllWindows()
