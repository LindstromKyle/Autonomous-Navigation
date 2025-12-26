import cv2
import numpy as np
from picamera2 import Picamera2
import time
from scipy.ndimage import label, find_objects

# --------------------- CONFIG ---------------------
PIXELS_PER_CM = 15.0  # <--- TUNE THIS!
MIN_FEATURES = 40  # Raised slightly for stability
REDETECT_EVERY_FRAMES = 20
MAX_CORNERS = 200  # Good stable number
# -------------------------------------------------

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()
time.sleep(2)

feature_params = dict(
    maxCorners=MAX_CORNERS, qualityLevel=0.3, minDistance=7, blockSize=7
)
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

old_frame = picam2.capture_array()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

# Preprocessing
nbins = 256
clip_limit_normalized = 0.01
clip_limit = clip_limit_normalized * nbins
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

pos_x = pos_y = 0.0
frame_count = 0

print("\n=== Martian Rover Navigation ===")
print("Stable feature refresh and hazard detection enabled.\n")

while True:
    frame = picam2.capture_array()
    frame_gray_raw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Pre-processing
    frame_gray = clahe.apply(frame_gray_raw)
    # frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), 0)  # Optional

    frame_count += 1

    dx_px = dy_px = 0.0
    good_new = np.empty((0, 1, 2))

    if p0 is not None and len(p0) > 0:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 5:
                flow = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
                dx_px = np.median(flow[:, 0])
                dy_px = np.median(flow[:, 1])

    # Default: continue with tracked points
    p0 = good_new.reshape(-1, 1, 2) if good_new.size > 0 else None

    # Refresh: replace with fresh strong features if low or periodic
    if (
        p0 is None or len(p0) < MIN_FEATURES
    ) or frame_count % REDETECT_EVERY_FRAMES == 0:
        new_pts = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        if new_pts is not None:
            p0 = new_pts
            mask = np.zeros_like(old_frame)  # Clear trails on full refresh

    # Update position
    dx_cm = -dx_px / PIXELS_PER_CM
    dy_cm = dy_px / PIXELS_PER_CM
    pos_x += dx_cm
    pos_y += dy_cm

    # --- Hazard Detection (High density = rocks/hazards; sparse = safe) ---
    grid_size = 6
    height, width = frame_gray.shape
    cell_h, cell_w = height // grid_size, width // grid_size
    density_grid = np.zeros((grid_size, grid_size), dtype=int)

    if len(good_new) > 0:
        pts = good_new.reshape(-1, 2)
        for x, y in pts:
            grid_x = min(int(x // cell_w), grid_size - 1)
            grid_y = min(int(y // cell_h), grid_size - 1)
            density_grid[grid_y, grid_x] += 1

    hazard_thresh = 2

    hazard_mask = density_grid > hazard_thresh
    safe_mask = ~hazard_mask

    # Find largest safe zone using connected components
    if np.any(safe_mask):
        labeled_safe, num_safe = label(safe_mask)
        if num_safe > 0:
            safe_sizes = [np.sum(labeled_safe == i) for i in range(1, num_safe + 1)]
            largest_safe_id = np.argmax(safe_sizes) + 1

            # Get bounding box of largest safe blob
            safe_slice = find_objects(labeled_safe == largest_safe_id)[0]

            # Geometric center (in grid coordinates)
            geo_center_y = (safe_slice[0].start + safe_slice[0].stop - 1) // 2
            geo_center_x = (safe_slice[1].start + safe_slice[1].stop - 1) // 2

            # Pixel position of geometric center
            geo_px = (
                geo_center_x * cell_w + cell_w // 2,
                geo_center_y * cell_h + cell_h // 2,
            )

            # Now find the closest safe (non-hazard) grid cell to this center
            # Only consider cells that are actually in the safe blob
            safe_y, safe_x = np.where(labeled_safe == largest_safe_id)
            safe_coords = list(zip(safe_y, safe_x))

            # If no safe cells (edge case), fallback to center
            if not safe_coords:
                safe_center_px = None
                rel_dx_cm = rel_dy_cm = None
            else:
                # Find distances from geo center to each safe cell center
                cell_centers = [
                    (sx * cell_w + cell_w // 2, sy * cell_h + cell_h // 2)
                    for sy, sx in safe_coords
                ]
                distances = [
                    np.hypot(cx - geo_px[0], cy - geo_px[1]) for cx, cy in cell_centers
                ]
                closest_idx = np.argmin(distances)
                safe_center_px = cell_centers[closest_idx]

                # Relative position in cm for commands
                rel_dx_px = safe_center_px[0] - width // 2
                rel_dy_px = safe_center_px[1] - height // 2
                rel_dx_cm = rel_dx_px / PIXELS_PER_CM
                rel_dy_cm = rel_dy_px / PIXELS_PER_CM
        else:
            safe_center_px = None
            rel_dx_cm = rel_dy_cm = None
    else:
        safe_center_px = None
        rel_dx_cm = rel_dy_cm = None

    # Draw trails and points
    for new, old in zip(good_new, good_old if "good_old" in locals() else []):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)

    # Overlay hazards (red) and safe zone (green circle)
    for gy in range(grid_size):
        for gx in range(grid_size):
            if density_grid[gy, gx] > hazard_thresh:
                top_left = (gx * cell_w, gy * cell_h)
                bottom_right = ((gx + 1) * cell_w, (gy + 1) * cell_h)
                cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    if safe_center_px is not None:
        cv2.circle(img, safe_center_px, 30, (0, 255, 0), 3)  # Larger green circle

    # Display info
    cv2.putText(
        img,
        f"Pos: ({pos_x:+.1f}, {pos_y:+.1f}) cm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"Features: {len(good_new)}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"Hazards: {np.sum(hazard_mask)}/{grid_size**2}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        "r = reset pos | q = quit",
        (10, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    cv2.imshow("Martian Rover Navigation", img)

    # Console commands toward safe zone
    if frame_count % 30 == 0:
        mag = abs(dx_cm) + abs(dy_cm)
        if mag > 0.5:
            dist = mag * 15
            dir_x = "right" if dx_cm > 0 else "left"
            dir_y = "forward" if dy_cm > 0 else "backward"
            direction = dir_x if abs(dx_cm) > abs(dy_cm) else dir_y
            print(f"→ Move {direction} {dist:.0f} cm")

        if rel_dx_cm is not None and rel_dy_cm is not None:
            # Safe zone command
            if abs(rel_dx_cm) > 5 or abs(rel_dy_cm) > 5:
                dir_x = "right" if rel_dx_cm > 0 else "left"
                dir_y = "forward" if rel_dy_cm > 0 else "backward"
                main_dir = dir_x if abs(rel_dx_cm) > abs(rel_dy_cm) else dir_y
                dist = max(abs(rel_dx_cm), abs(rel_dy_cm))
                print(f"→ Safe landing zone: Move {main_dir} {dist:.0f} cm")
            else:
                print("→ Current position is safe for landing")
                pass
        else:
            print("No safe landing zone detected")

    old_gray = frame_gray.copy()

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        pos_x = pos_y = 0.0
        print("Position reset")

picam2.stop()
cv2.destroyAllWindows()
