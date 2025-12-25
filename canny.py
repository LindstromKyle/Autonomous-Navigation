import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
import cv2


def apply_power_law(image, gamma, c=1.0):
    """
    Applies a power law (gamma) transformation to an image.

    Args:
        image.
        gamma (float): The gamma value for the power law transformation.
        c (float): The scaling constant (default is 1.0).

    Returns:
        numpy.ndarray: The transformed image.
    """

    # Normalize pixel intensities to [0, 1]
    normalized_img = image / 255.0

    # Apply power law transformation
    transformed_img = c * np.power(normalized_img, gamma)

    # Scale back to [0, 255] and convert to uint8
    transformed_img = np.array(transformed_img * 255, dtype=np.uint8)

    return transformed_img


# Step 1: Load the image and convert to grayscale explicitly
# Open the JPEG file, convert to grayscale ('L' mode), and turn into a NumPy array (0-255, uint8)
image_path = "/home/kyle/repos/EDL/test_image.jpg"  # Replace with your JPEG file path
img = Image.open(image_path)

manual = False

if manual:

    gray_img = np.array(
        img.convert("L")
    )  # Explicit conversion: averages RGB channels if color image
    print(f"Grayscale image shape: {gray_img.shape}, dtype: {gray_img.dtype}")

    # Optional: Visualize grayscale image
    plt.imshow(gray_img, cmap="gray")
    plt.title("Step 1: Grayscale Image")
    plt.show()

    # Step 2: Apply Gaussian filter for noise reduction
    # Use a 5x5 Gaussian kernel (sigma=1.4 is common for Canny; adjust as needed)
    # We normalize the kernel to sum to 1
    def gaussian_kernel(size=5, sigma=1.4):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)
                / (2 * sigma**2)
            ),
            (size, size),
        )
        return kernel / np.sum(kernel)

    gaussian = gaussian_kernel()
    smoothed_img = convolve2d(
        gray_img, gaussian, mode="same", boundary="symm"
    )  # Convolve to blur
    smoothed_img = np.clip(smoothed_img, 0, 255)  # Clip to valid range

    # Optional: Visualize smoothed image
    plt.imshow(smoothed_img, cmap="gray")
    plt.title("Step 2: Gaussian Smoothed Image")
    plt.show()

    # Step 3: Apply Sobel operators in x and y to compute gradients
    # Sobel kernels for horizontal (x) and vertical (y) edges
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Convolve to get gradients
    grad_x = convolve2d(smoothed_img, sobel_x, mode="same", boundary="symm")
    grad_y = convolve2d(smoothed_img, sobel_y, mode="same", boundary="symm")

    # Compute gradient magnitude and angle explicitly
    magnitude = np.hypot(grad_x, grad_y)  # sqrt(grad_x^2 + grad_y^2)
    magnitude = magnitude / magnitude.max() * 255  # Normalize to 0-255
    angle = np.arctan2(grad_y, grad_x)  # Angle in radians (-pi to pi)
    angle = np.degrees(angle)  # Convert to degrees (-180 to 180)
    angle[angle < 0] += 180  # Map to 0-180 for easier direction binning

    # Optional: Visualize magnitude
    plt.imshow(magnitude, cmap="gray")
    plt.title("Step 3: Gradient Magnitude")
    plt.show()

    # Step 4: Non-Maximum Suppression (NMS) for edge thinning
    # Create an output array initialized to zero
    nms_img = np.zeros(magnitude.shape, dtype=np.uint8)

    # Bin angles into 4 directions: 0 (horizontal), 45, 90 (vertical), 135 degrees
    angle_quantized = np.round(angle / 45) * 45
    angle_quantized[angle_quantized == 180] = 0  # Wrap 180 to 0

    # Iterate over each pixel (explicit loop for clarity; avoid borders)
    for i in range(1, magnitude.shape[0] - 1):
        for j in range(1, magnitude.shape[1] - 1):
            # Get neighbors based on angle direction
            if angle_quantized[i, j] == 0:  # Horizontal
                neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
            elif angle_quantized[i, j] == 45:  # Diagonal /
                neighbors = [magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]]
            elif angle_quantized[i, j] == 90:  # Vertical
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
            elif angle_quantized[i, j] == 135:  # Diagonal \
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

            # If current pixel is greater than both neighbors, keep it; else suppress
            if magnitude[i, j] >= max(neighbors):
                nms_img[i, j] = magnitude[i, j]

    # Optional: Visualize NMS result
    plt.imshow(nms_img, cmap="gray")
    plt.title("Step 4: After Non-Maximum Suppression")
    plt.show()

    # Step 5: Double thresholding
    # Define low and high thresholds (common defaults; adjust based on your image, e.g., 30 and 100)
    low_threshold = 50
    high_threshold = 150

    # Classify: strong (> high), weak (between low and high), non-edge (< low)
    strong_edges = nms_img > high_threshold
    weak_edges = (nms_img >= low_threshold) & (nms_img <= high_threshold)
    edges = np.zeros(nms_img.shape, dtype=np.uint8)
    edges[strong_edges] = 255  # Strong edges start as definite

    # Step 6: Hysteresis edge tracking
    # Use a stack-based approach to connect weak edges to strong ones (like flood fill but for edges)
    # Find all strong edge positions
    strong_i, strong_j = np.where(strong_edges)

    # Stack for positions to check
    stack = list(zip(strong_i, strong_j))

    # 8-connected neighbors
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while stack:
        i, j = stack.pop()
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (0 <= ni < edges.shape[0]) and (0 <= nj < edges.shape[1]):
                if (
                    weak_edges[ni, nj] and edges[ni, nj] == 0
                ):  # If weak and not yet marked
                    edges[ni, nj] = 255  # Promote to strong
                    stack.append((ni, nj))  # Add to stack to check its neighbors

    # Optional: Visualize final edges
    plt.imshow(edges, cmap="gray")
    plt.title("Step 6: Final Edges After Hysteresis")
    plt.show()

    # Save the final edge image (optional)
    edge_img = Image.fromarray(edges)
    edge_img.save("edges.jpg")

    # If you want post-processing (your steps 7-9), add here:
    # 7. Morphological close: Dilation followed by erosion (requires scipy.ndimage)
    # from scipy.ndimage import binary_dilation, binary_erosion
    # kernel = np.ones((3, 3), np.uint8)  # Small structuring element
    # closed_edges = binary_erosion(binary_dilation(edges > 0, structure=kernel), structure=kernel)
    # closed_edges = closed_edges.astype(np.uint8) * 255

    # 8. Fill (e.g., flood fill from background or use binary fill holes)
    # from scipy.ndimage import binary_fill_holes
    # filled = binary_fill_holes(closed_edges > 0).astype(np.uint8) * 255

    # 9. Segment (e.g., find contours or label regions)
    # from skimage.measure import label, regionprops  # If scikit-image installed
    # labels = label(filled > 0)
    # for region in regionprops(labels):s
    #     print(f"Region area: {region.area}")  # Example: Analyze segments for safe zones

else:
    grey_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # 2. Downscale early (use area interpolation)
    scale = 0.33  # ~640x360
    small_img = cv2.resize(
        grey_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    # 4. Optional: blur to reduce wood grain noise
    # gray = cv2.GaussianBlur(gray, (5,5), 0)

    preprocessed_img = small_img

    transformed_img = preprocessed_img
    # transformed_img = apply_power_law(preprocessed, 0.5, 1)
    # transformed_img = cv2.equalizeHist(preprocessed_img)

    preprocessed_pix_vals = preprocessed_img.flatten()
    transformed_pix_vals = transformed_img.flatten()

    #     fig, ax = plt.subplots(2, 2, figsize=(14,8))
    #     ax[0][0].imshow(preprocessed_img, cmap="grey")
    #     ax[1][0].imshow(transformed_img, cmap="grey")
    #     ax[0][1].hist(preprocessed_pix_vals, bins=256, range=(0, 256))
    #     ax[1][1].hist(transformed_pix_vals, bins=256, range=(0, 256))

    # pixel_values = grey.ravel()
    # ax.hist(pixel_values, bins=256, range=(0, 256))
    plt.show()

    edges = cv2.Canny(transformed_img, 50, 150)  # Adjust thresholds as needed

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 6. Find contours (hazards = closed blobs)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. Filter by size/area (rocks vs noise)
    min_rock_area = 50  # tune this
    hazards = [c for c in contours if cv2.contourArea(c) > min_rock_area]
    contour_vis = preprocessed_img.copy()
    cv2.drawContours(contour_vis, hazards, -1, (0, 255, 0), 2)  # green

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    ax[0][0].imshow(preprocessed_img, cmap="grey")
    ax[0][0].set_title("Original Image")
    ax[0][1].imshow(edges, cmap="grey")
    ax[0][1].set_title("Canny Edges")
    ax[1][0].imshow(closed, cmap="grey")
    ax[1][0].set_title("Closed")
    ax[1][1].imshow(contour_vis, cmap="grey")
    ax[1][1].set_title("Contours")
    plt.tight_layout()
    plt.show()
    # plt.imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))

    #
    #     # 8. Safe landing: find largest empty rectangle or grid cell
    #     # Create binary hazard map
    #     hazard_map = np.zeros(preprocessed_img.shape[:2], dtype=np.uint8)
    #     cv2.drawContours(hazard_map, hazards, -1, 255, thickness=cv2.FILLED)
    #
    #     # Divide into grid (e.g., 6x6)
    #     h, w = hazard_map.shape
    #     grid_h, grid_w = h//6, w//6
    #     safe_spots = []
    #
    #     for i in range(6):
    #         for j in range(6):
    #             cell = hazard_map[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
    #             if cv2.countNonZero(cell) == 0:  # fully clear
    #                 safe_spots.append((i, j))
    #
    #     # Map safe spot back to full-res
    #     safe_i, safe_j = safe_spots[0]  # pick best
    #     center_x = int((safe_j + 0.5) * grid_w / scale)
    #     center_y = int((safe_i + 0.5) * grid_h / scale)
    #     cv2.circle(grey_img, (center_x, center_y), 50, (0,255,0), 3)
    #     cv2.imshow("Landing Zone", grey_img)
