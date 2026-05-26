# ---------------------------------
# MAINTENANCE RECOMMENDATION
# ---------------------------------

def get_recommendation(
    prediction,
    severity
):

    # ---------------------------------
    # HEALTHY
    # ---------------------------------

    if prediction == "Healthy":

        if severity < 10:

            return (
                "Bearing operating normally. "
                "Continue routine monitoring."
            )

        else:

            return (
                "Minor thermal variation detected. "
                "Schedule preventive inspection."
            )

    # ---------------------------------
    # OLD USED
    # ---------------------------------

    elif prediction == "Old Used Bearing":

        return (
            "Bearing wear detected. "
            "Lubrication and vibration inspection recommended."
        )

    # ---------------------------------
    # IRON SCRAP
    # ---------------------------------

    elif prediction == "Iron Scrap Bearing":

        return (
            "Severe bearing damage detected. "
            "Immediate replacement recommended."
        )

    # ---------------------------------
    # BALL DEFECT
    # ---------------------------------

    elif prediction == "Ball Defect Bearing":

        return (
            "Ball defect detected. "
            "Check rolling elements and replace bearing immediately."
        )

    # ---------------------------------
    # DEFAULT
    # ---------------------------------

    else:

        return (
            "Further inspection required."
        )
