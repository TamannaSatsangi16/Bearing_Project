import cv2
import numpy as np

# ---------------------------------
# PROCESS IMAGE
# ---------------------------------

def process_image(img):

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_GRAY2RGB
        )

    else:

        img_rgb = img

    # Convert to grayscale
    gray = cv2.cvtColor(
        img_rgb,
        cv2.COLOR_RGB2GRAY
    )

    # Edge detection
    edges = cv2.Canny(
        gray,
        100,
        200
    )

    return img_rgb, gray, edges


# ---------------------------------
# DYNAMIC HOTSPOT DETECTION
# ---------------------------------

def detect_hotspots(gray, sensitivity=1.2):

    # Dynamic threshold
    mean_val = np.mean(gray)

    std_val = np.std(gray)

    dynamic_threshold = (
        mean_val + sensitivity * std_val
    )

    # Apply threshold
    _, hotspot = cv2.threshold(
        gray,
        dynamic_threshold,
        255,
        cv2.THRESH_BINARY
    )

    return hotspot, dynamic_threshold