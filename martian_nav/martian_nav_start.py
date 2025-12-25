import cv2
import numpy as np
from picamera2 import Picamera2
import time
import matplotlib.pyplot as plt

# --------------------- CONFIG ---------------------
PIXELS_PER_CM = 20.0  # <--- TUNE THIS!
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

# fig = plt.figure()

# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
# ax4 = fig.add_subplot(2, 2, 4, sharex=ax2)

# ax1.imshow(old_gray)

# ax2.hist(old_gray.ravel(), bins=256, range=(0, 255))

# transformed = clahe.apply(old_gray)
# ax3.imshow(transformed)
# ax4.hist(transformed.ravel(), bins=256, range=(0, 255))

# plt.tight_layout()
# plt.show()


pos_x = pos_y = 0.0
frame_count = 0

print("\n=== Martian Rover Navigation ===")
print("Stable feature refresh enabled.\n")

while True:
    frame = picam2.capture_array()
    frame_gray_raw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Pre-processing

    frame_gray = clahe.apply(frame_gray_raw)
    # frame_gray = cv2.GaussianBlur(frame_gray, (3, 3), 0)  # Optional, but helps

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
            print(f"Feature refresh → {len(p0)} new strong features")

    # Update position
    dx_cm = -dx_px / PIXELS_PER_CM
    dy_cm = dy_px / PIXELS_PER_CM
    pos_x += dx_cm
    pos_y += dy_cm

    # Draw only successfully tracked trails
    for new, old in zip(good_new, good_old if "good_old" in locals() else []):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)

    # Clean display: show only current tracked count
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
        "r = reset pos | q = quit",
        (10, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    cv2.imshow("Martian Rover Navigation", img)

    # Commands
    if frame_count % 10 == 0:
        mag = abs(dx_cm) + abs(dy_cm)
        if mag > 0.5:
            dist = mag * 15
            dir_x = "right" if dx_cm > 0 else "left"
            dir_y = "forward" if dy_cm > 0 else "backward"
            direction = dir_x if abs(dx_cm) > abs(dy_cm) else dir_y
            print(f"→ Move {direction} {dist:.0f} cm")

    old_gray = frame_gray.copy()

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        pos_x = pos_y = 0.0
        print("Position reset")

picam2.stop()
cv2.destroyAllWindows()
