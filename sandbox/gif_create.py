import h5py
import cv2
import numpy as np
from PIL import Image
import os

H5_PATH = "/home/kyle/recordings/recording_20260201_203336.h5"

# Frame range to include in the GIF
START_FRAME = 0  # inclusive
END_FRAME = 249  # exclusive (None = all remaining frames)

# If no timestamps exist in the file, use this constant FPS
FALLBACK_FPS = 12.0

# Minimum frame duration in milliseconds (prevents 0 ms or negative values)
MIN_DURATION_MS = 20

# Output GIF filename (will be saved in the same folder as the .h5 by default)
# Set to None to auto-generate name like "recording_20260201_192831.gif"
OUTPUT_GIF = None

# Optional: resize the output GIF frames (set to None to keep original size)
# Example: OUTPUT_SIZE = (320, 240)  # width, height
OUTPUT_SIZE = None  # or e.g. (480, 360), (640, 480), etc.

# ────────────────────────────────────────────────


def main():
    print(f"Loading {H5_PATH}...")
    try:
        with h5py.File(H5_PATH, "r") as f:
            if "frames" not in f:
                print("Error: No 'frames' dataset found in file.")
                return
            frames_slice = f["frames"][START_FRAME:END_FRAME]
            if "timestamps" in f:
                ts_slice = f["timestamps"][START_FRAME:END_FRAME]
            else:
                ts_slice = None
                print("Warning: No timestamps found — using constant FPS.")
    except Exception as e:
        print(f"Failed to open HDF5 file: {e}")
        return

    if len(frames_slice) == 0:
        print("No frames selected.")
        return

    n_frames = len(frames_slice)
    print(f"Selected {n_frames} frames (indices {START_FRAME} to {END_FRAME or 'end'})")

    # Compute durations in milliseconds
    durations_ms = []
    if ts_slice is not None and len(ts_slice) > 1:
        diffs = np.diff(ts_slice)
        durations_ms = [max(MIN_DURATION_MS, int(d * 1000)) for d in diffs]
        # Last frame gets average duration (or fallback)
        avg_dur = int(np.mean(durations_ms)) if durations_ms else MIN_DURATION_MS
        durations_ms.append(avg_dur)
        print("Using variable frame durations from timestamps.")
    else:
        frame_dur_ms = int(1000 / FALLBACK_FPS)
        durations_ms = [frame_dur_ms] * n_frames
        print(
            f"Using constant duration: {frame_dur_ms} ms per frame ({FALLBACK_FPS:.1f} fps)"
        )

    # Prepare output path
    if OUTPUT_GIF is None:
        base = os.path.splitext(os.path.basename(H5_PATH))[0]
        output_gif = os.path.join(os.path.dirname(H5_PATH), f"{base}.gif")
    else:
        output_gif = OUTPUT_GIF

    print(f"Writing GIF → {output_gif}")

    # Convert frames to PIL (BGR → RGB)
    pil_images = []
    for i, frame in enumerate(frames_slice):
        print(f"Preparing frame {i+1} of {len(frames_slice)}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if OUTPUT_SIZE is not None:
            frame_rgb = cv2.resize(frame_rgb, OUTPUT_SIZE)
        pil_images.append(Image.fromarray(frame_rgb))

    if not pil_images:
        print("No images to save.")
        return

    # Save animated GIF
    print("Generating GIF")
    pil_images[0].save(
        output_gif,
        save_all=True,
        append_images=pil_images[1:],
        duration=durations_ms,
        loop=0,  # 0 = infinite loop
        optimize=True,  # helps reduce file size
        disposal=2,  # recommended for clean animation
    )

    print(f"GIF created successfully with {n_frames} frames.")
    print(f"Total real-time duration ≈ {sum(durations_ms)/1000:.1f} seconds")


if __name__ == "__main__":
    main()
