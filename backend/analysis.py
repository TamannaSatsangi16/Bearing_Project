import numpy as np

# ---------------------------------
# ANALYZE IMAGE
# ---------------------------------

def analyze_image(gray, hotspot):

    avg_intensity = np.mean(gray)

    max_intensity = np.max(gray)

    min_intensity = np.min(gray)

    # Hotspot pixels
    hotspot_pixels = np.sum(hotspot == 255)

    total_pixels = hotspot.size

    hotspot_percentage = (
        hotspot_pixels / total_pixels
    ) * 100

    severity = hotspot_percentage

    # Fault logic
    if hotspot_percentage > 25:

        status = "OVERHEATING DETECTED"

    elif hotspot_percentage > 10:

        status = "MODERATE HEATING"

    else:

        status = "NORMAL"

    return {

        "avg_intensity": avg_intensity,

        "max_intensity": max_intensity,

        "min_intensity": min_intensity,

        "severity": severity,

        "hotspot_percentage": hotspot_percentage,

        "status": status
    }
