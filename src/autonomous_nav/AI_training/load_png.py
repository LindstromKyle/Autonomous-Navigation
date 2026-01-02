import cv2
import numpy as np
from pathlib import Path


def load_png_to_numpy(image_path: str | Path) -> np.ndarray:
    """
    Load a PNG image from file and return it as a NumPy array.

    Args:
        image_path (str or Path): Path to the PNG file.

    Returns:
        np.ndarray: Image as a NumPy array.
                    - Shape: (height, width, channels) for color images (BGR order with OpenCV)
                    - Shape: (height, width) for grayscale images
                    - dtype: uint8 typically

    Raises:
        FileNotFoundError: If the image file doesn't exist.
        ValueError: If the image cannot be loaded (e.g., corrupted).
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # cv2.IMREAD_COLOR loads as BGR (default)
    # Use cv2.IMREAD_UNCHANGED to preserve alpha channel if present
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(
            f"Failed to load image (possibly corrupted or unsupported format): {image_path}"
        )

    return img


# ================ Example Usage ================
if __name__ == "__main__":
    # Replace with your actual PNG file path
    png_file = "/home/kyle/marsdata_v2/yolo_dataset_grayscale/images/test/332_0332ML0013430000107963E01_DXXX_tup.png"  # or "/path/to/your/image.png"

    try:
        image_array = load_png_to_numpy(png_file)
        print(f"Successfully loaded image!")
        print(f"Shape: {image_array.shape}")
        print(f"Data type: {image_array.dtype}")
        print(f"Pixel value range: {image_array.min()} to {image_array.max()}")

        # Optional: Convert BGR to RGB if you plan to use with matplotlib or PIL
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            print("Converted to RGB for display/plotting.")

        # Optional: Display with matplotlib
        # import matplotlib.pyplot as plt
        # plt.imshow(image_rgb if 'image_rgb' in locals() else image_array, cmap='gray')
        # plt.axis('off')
        # plt.show()

    except Exception as e:
        print(f"Error: {e}")
