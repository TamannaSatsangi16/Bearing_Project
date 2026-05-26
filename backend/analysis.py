import numpy as np

# ---------------------------------
# ANALYZE IMAGE
# ---------------------------------

def analyze_image(
    gray,
    hotspot
):

    # Average intensity
    avg_intensity = np.mean(
        gray
    )

    # Maximum intensity
    max_intensity = np.max(
        gray
    )

    # Minimum intensity
    min_intensity = np.min(
        gray
    )

    # Hotspot area
    hotspot_pixels = np.sum(
        hotspot > 0
    )

    total_pixels = hotspot.size

    hotspot_percentage = (
        hotspot_pixels / total_pixels
    ) * 100

    severity = hotspot_percentage

    # ---------------------------------
    # STATUS
    # ---------------------------------

    if severity > 25:

        status = "OVERHEATING DETECTED"

    elif severity > 10:

        status = "MODERATE HEATING"

    else:

        status = "NORMAL"

    # ---------------------------------
    # THERMAL CONDITION
    # ---------------------------------

    if avg_intensity < 80:

        thermal_condition = "COLD"

    elif avg_intensity < 140:

        thermal_condition = "NORMAL"

    elif avg_intensity < 190:

        thermal_condition = "HOT"

    else:

        thermal_condition = "RED HOT / CRITICAL"

    # ---------------------------------
    # RETURN RESULTS
    # ---------------------------------

    return {

        "avg_intensity": avg_intensity,

        "max_intensity": max_intensity,

        "min_intensity": min_intensity,

        "severity": severity,

        "hotspot_percentage": hotspot_percentage,

        "status": status,

        "thermal_condition": thermal_condition
    }
