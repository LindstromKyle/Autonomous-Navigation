# Utility functions


def pixels_to_cm(pixels: float, height: float, focal_len: float) -> float:
    """
    Converts pixels to cm based on camera height
    """
    if height <= 0:
        return 0.0
    return pixels * height / focal_len


def cm_to_pixels(cm: float, height: float, focal_len: float) -> float:
    """
    Converts cm to pixels based on camera height
    """
    if height <= 0:
        return 0.0
    return cm * focal_len / height
