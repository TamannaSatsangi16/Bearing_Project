# ---------------------------------
# MAINTENANCE RECOMMENDATIONS
# ---------------------------------

def get_recommendation(status):

    if status == "OVERHEATING DETECTED":

        return (
            "Inspect lubrication and cooling system immediately"
        )

    elif status == "MODERATE HEATING":

        return (
            "Monitor bearing condition regularly"
        )

    else:

        return (
            "Bearing condition is healthy"
        )