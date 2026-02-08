from PIL import Image

# Open image
img = Image.open("/home/kyle/repos/Autonomous-Navigation/readme_imgs/hardware.jpeg")

# Rotate (counter-clockwise) — common angles: 90, 180, 270, -90, etc.
# expand=True → enlarges canvas so corners aren't cut off
# img = img.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)

# Now crop (left, upper, right, lower)
crop_box = (400, 0, 1400, 1070)  # adjust to your needs
cropped = img.crop(crop_box)

# Save with good quality
cropped.save(
    "/home/kyle/repos/Autonomous-Navigation/readme_imgs/rotated_cropped.jpeg",
    quality=92,
)
