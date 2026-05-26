import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# ---------------------------------
# BACKEND IMPORTS
# ---------------------------------

from backend.image_processing import (
    process_image,
    detect_hotspots,
    detect_edges
)

from backend.analysis import (
    analyze_image
)

from backend.predict import (
    predict_bearing
)

from backend.recommendations import (
    get_recommendation
)

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="AI Bearing Analyzer",
    layout="wide"
)

# ---------------------------------
# TITLE
# ---------------------------------

st.title(
    "AI Bearing Maintenance Analyzer"
)

st.write(
    "Predictive Thermal Monitoring System"
)

# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title(
    "System Controls"
)

sensitivity = st.sidebar.slider(
    "Hotspot Sensitivity",
    0.5,
    3.0,
    1.2
)

# ---------------------------------
# FILE UPLOADER
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload Bearing Thermal Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# PROCESS IMAGE
# ---------------------------------

if uploaded_file is not None:

    # Read image
    image = Image.open(
        uploaded_file
    )

    image_np = np.array(
        image
    )

    # ---------------------------------
    # IMAGE PROCESSING
    # ---------------------------------

    gray = process_image(
        image_np
    )

    edges = detect_edges(
        gray
    )

    hotspot, dynamic_threshold = detect_hotspots(
        gray,
        sensitivity
    )

    # ---------------------------------
    # ANALYSIS
    # ---------------------------------

    results = analyze_image(
        gray,
        hotspot
    )

    # ---------------------------------
    # AI PREDICTION
    # ---------------------------------

    prediction, confidence = predict_bearing(
        image_np
    )

    # ---------------------------------
    # AI PREDICTION DISPLAY
    # ---------------------------------

    st.subheader(
        "AI Prediction"
    )

    st.success(
        f"Prediction Result: {prediction}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # ---------------------------------
    # ANALYSIS METRICS
    # ---------------------------------

    st.subheader(
        "Analysis Metrics"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Average Intensity",
        f"{results['avg_intensity']:.2f}"
    )

    col2.metric(
        "Maximum Intensity",
        f"{results['max_intensity']}"
    )

    col3.metric(
        "Minimum Intensity",
        f"{results['min_intensity']}"
    )

    col4.metric(
        "Severity %",
        f"{results['severity']:.2f}%"
    )

    col5.metric(
        "Dynamic Threshold",
        f"{dynamic_threshold:.2f}"
    )

    # ---------------------------------
    # THERMAL CONDITION
    # ---------------------------------

    st.subheader(
        "Thermal Condition"
    )

    condition = results[
        "thermal_condition"
    ]

    if condition == "COLD":

        st.info(
            condition
        )

    elif condition == "NORMAL":

        st.success(
            condition
        )

    elif condition == "HOT":

        st.warning(
            condition
        )

    else:

        st.error(
            condition
        )

    # ---------------------------------
    # SYSTEM STATUS
    # ---------------------------------

    st.subheader(
        "System Status"
    )

    if results["status"] == "NORMAL":

        st.success(
            results["status"]
        )

    elif results["status"] == "MODERATE HEATING":

        st.warning(
            results["status"]
        )

    else:

        st.error(
            results["status"]
        )

    # ---------------------------------
    # MAINTENANCE RECOMMENDATION
    # ---------------------------------

    recommendation = get_recommendation(
        prediction,
        results["severity"]
    )

    st.subheader(
        "Maintenance Recommendation"
    )

    st.info(
        recommendation
    )

    # ---------------------------------
    # IMAGE PROCESSING DISPLAY
    # ---------------------------------

    st.subheader(
        "Image Processing"
    )

    col6, col7, col8 = st.columns(3)

    with col6:

        st.image(
            gray,
            caption="Grayscale",
            use_container_width=True
        )

    with col7:

        st.image(
            edges,
            caption="Edge Detection",
            use_container_width=True
        )

    with col8:

        st.image(
            hotspot,
            caption="Hotspot Detection",
            use_container_width=True
        )

    # ---------------------------------
    # HISTOGRAM
    # ---------------------------------

    st.subheader(
        "Pixel Intensity Histogram"
    )

    fig, ax = plt.subplots()

    ax.hist(
        gray.ravel(),
        bins=50
    )

    ax.set_xlabel(
        "Pixel Intensity"
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(
        fig
    )

    # ---------------------------------
    # SAVE HISTORY
    # ---------------------------------

    history_entry = {

        "Image": uploaded_file.name,

        "Prediction": prediction,

        "Confidence": confidence,

        "Severity": results["severity"],

        "Status": results["status"]
    }

    history_df = pd.DataFrame(
        [history_entry]
    )

    if os.path.exists(
        "history.csv"
    ):

        old_history = pd.read_csv(
            "history.csv"
        )

        updated_history = pd.concat(
            [old_history, history_df],
            ignore_index=True
        )

        updated_history.to_csv(
            "history.csv",
            index=False
        )

    else:

        history_df.to_csv(
            "history.csv",
            index=False
        )

# ---------------------------------
# ANALYSIS HISTORY
# ---------------------------------

st.subheader(
    "Analysis History"
)

if os.path.exists(
    "history.csv"
):

    history_data = pd.read_csv(
        "history.csv"
    )

    # ---------------------------------
    # DELETE ALL BUTTON
    # ---------------------------------

    if st.button(
        "🗑️ Delete All History"
    ):

        empty_df = pd.DataFrame(columns=[
            "Image",
            "Prediction",
            "Confidence",
            "Severity",
            "Status"
        ])

        empty_df.to_csv(
            "history.csv",
            index=False
        )

        st.rerun()

    st.write("---")

    # ---------------------------------
    # TABLE HEADER
    # ---------------------------------

    h1, h2, h3, h4, h5, h6 = st.columns(
        [2,2,2,2,2,1]
    )

    h1.write("Image")
    h2.write("Prediction")
    h3.write("Confidence")
    h4.write("Severity")
    h5.write("Status")
    h6.write("Delete")

    st.write("---")

    # ---------------------------------
    # SHOW ROWS
    # ---------------------------------

    for index, row in history_data.iterrows():

        c1, c2, c3, c4, c5, c6 = st.columns(
            [2,2,2,2,2,1]
        )

        c1.write(
            row["Image"]
        )

        c2.write(
            row["Prediction"]
        )

        c3.write(
            row["Confidence"]
        )

        c4.write(
            row["Severity"]
        )

        c5.write(
            row["Status"]
        )

        # ---------------------------------
        # DELETE BUTTON
        # ---------------------------------

        if c6.button(
            "🗑️",
            key=f"delete_{index}"
        ):

            history_data = history_data.drop(
                index
            )

            history_data = history_data.reset_index(
                drop=True
            )

            history_data.to_csv(
                "history.csv",
                index=False
            )

            st.rerun()
