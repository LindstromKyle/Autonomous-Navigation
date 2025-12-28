from picamera2 import Picamera2
import numpy as np
import matplotlib.pyplot as plt

# Select name
image_name = "/home/kyle/repos/Autonomous-Navigation/images/calibration.jpg"

# Initialize the camera
picam2 = Picamera2()

# Configure for still capture (adjust resolution if needed, e.g., for lower res: {'size': (640, 480)})
# config = picam2.create_still_configuration(main={"size": (1920, 1080)})
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)

# Start the camera
picam2.start()

# Capture a single image as a NumPy array (RGB format by default)
image_array = picam2.capture_array()

# Stop the camera
picam2.stop()

# Inspect the array structure
print("Image shape:", image_array.shape)
print("Data type:", image_array.dtype)

plt.imshow(image_array)
plt.show()

# Optionally, save it to a file for visual check (requires OpenCV: pip install opencv-python)
import cv2

cv2.imwrite(image_name, image_array)
