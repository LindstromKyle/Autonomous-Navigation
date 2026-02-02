import h5py
import cv2
import numpy as np  # good to have even if not used yet

h5_path = "/home/kyle/recordings/recording_20260201_203336.h5"
print(f"Loading {h5_path}...")
try:
    with h5py.File(h5_path, "r") as f:
        if "frames" not in f:
            print("Error: No 'frames' dataset found in file.")
            exit(1)
        frames = f["frames"][:]
        if "timestamps" in f:
            timestamps = f["timestamps"][:]
        else:
            timestamps = None
            print("Warning: No timestamps found — showing frame indices only.")
except Exception as e:
    print(f"Failed to open HDF5 file: {e}")
    exit(1)

if len(frames) == 0:
    print("No frames in the file.")
    exit(0)

total_frames = len(frames)
print(f"Loaded {total_frames} frames (shape: {frames.shape})")

current_idx = 0

# Create the window once, before the loop
cv2.namedWindow("HDF5 Frame Viewer", cv2.WINDOW_NORMAL)

while True:
    frame = frames[current_idx]

    # Optional resize for display
    display_frame = frame
    if frame.shape[0] > 800 or frame.shape[1] > 1200:
        scale = min(800 / frame.shape[0], 1200 / frame.shape[1])
        display_frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # Build title
    ts_str = ""
    if timestamps is not None and current_idx < len(timestamps):
        ts_str = f" | t={timestamps[current_idx]:.2f}s"

    title = f"Frame {current_idx+1}/{total_frames}{ts_str}  (← → arrows | PgUp/PgDn ±10 | Home/End | 0-9 jump % | q=quit)"

    # Set title AFTER window exists
    cv2.setWindowTitle("HDF5 Frame Viewer", title)

    cv2.imshow("HDF5 Frame Viewer", display_frame)

    key = cv2.waitKey(0) & 0xFF

    if key in (ord("q"), 27):  # q or ESC
        break

    elif key == 81 or key == ord("a"):  # left / A
        current_idx = max(0, current_idx - 1)

    elif key == 83 or key == ord("d"):  # right / D
        current_idx = min(total_frames - 1, current_idx + 1)

    elif key == 82:  # Page Up
        current_idx = max(0, current_idx - 10)

    elif key == 84:  # Page Down
        current_idx = min(total_frames - 1, current_idx + 10)

    elif key == 80:  # Home
        current_idx = 0

    elif key == 87:  # End
        current_idx = total_frames - 1

    # 0-9 → percentage jump
    elif ord("0") <= key <= ord("9"):
        percent = (key - ord("0")) / 9.0
        current_idx = int(percent * (total_frames - 1))

cv2.destroyAllWindows()
print("Done.")
