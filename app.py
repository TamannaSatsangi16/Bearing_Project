import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime
import os

# ---------------------------------
# BACKEND IMPORTS
# ---------------------------------

from backend.image_processing import (
    process_image,
    detect_hotspots
)

from backend.analysis import (
    analyze_image
)

from backend.recommendations import (
    get_recommendation
)

from backend.predict import (
    predict_bearing
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
# MAIN PROCESSING
# ---------------------------------

if uploaded_file is not None:

    # ---------------------------------
    # LOAD IMAGE
    # ---------------------------------

    image = Image.open(
        uploaded_file
    )

    img = np.array(
        image
    )

    # ---------------------------------
    # CREATE UPLOADS FOLDER
    # ---------------------------------

    os.makedirs(
        "uploads",
        exist_ok=True
    )

    image_path = f"uploads/{uploaded_file.name}"

    with open(
        image_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    # ---------------------------------
    # IMAGE PROCESSING
    # ---------------------------------

    img_rgb, gray, edges = process_image(
        img
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
        img_rgb
    )

    # ---------------------------------
    # MAINTENANCE RECOMMENDATION
    # ---------------------------------

    recommendation = get_recommendation(
        prediction,
        results["severity"]
    )

    # ---------------------------------
    # SAVE HISTORY
    # ---------------------------------

    history = {

        "Timestamp": datetime.now(),

        "Image": uploaded_file.name,

        "Prediction": prediction,

        "Confidence": confidence,

        "Severity": results["severity"],

        "Status": results["status"]
    }

    history_df = pd.DataFrame(
        [history]
    )

    if os.path.exists(
        "history.csv"
    ):

        history_df.to_csv(
            "history.csv",
            mode='a',
            header=False,
            index=False
        )

    else:

        history_df.to_csv(
            "history.csv",
            index=False
        )

    # ---------------------------------
    # ORIGINAL IMAGE
    # ---------------------------------

    st.subheader(
        "Original Thermal Image"
    )

    st.image(
        img_rgb,
        use_container_width=True
    )

    # ---------------------------------
    # AI PREDICTION
    # ---------------------------------

    st.subheader(
        "AI Prediction"
    )

    st.write(
        f"Prediction Result: {prediction}"
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )

    # ---------------------------------
    # PREDICTION STATUS
    # ---------------------------------

    if prediction == "Healthy":

        st.success(
            "HEALTHY BEARING"
        )

    elif prediction == "Old Used Bearing":

        st.warning(
            "OLD USED BEARING DETECTED"
        )

    elif prediction == "Iron Scrap Bearing":

        st.error(
            "IRON SCRAP / DAMAGED BEARING DETECTED"
        )

    elif prediction == "Ball Defect Bearing":

        st.error(
            "BALL DEFECT DETECTED"
        )

    # ---------------------------------
    # ANALYSIS METRICS
    # ---------------------------------

    st.subheader(
        "Analysis Metrics"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Average Intensity",
            f"{results['avg_intensity']:.2f}"
        )

    with col2:

        st.metric(
            "Maximum Intensity",
            f"{results['max_intensity']}"
        )

    with col3:

        st.metric(
            "Minimum Intensity",
            f"{results['min_intensity']}"
        )

    with col4:

        st.metric(
            "Severity %",
            f"{results['severity']:.2f}%"
        )

    with col5:

        st.metric(
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

    if results["status"] == "OVERHEATING DETECTED":

        st.error(
            results["status"]
        )

    elif results["status"] == "MODERATE HEATING":

        st.warning(
            results["status"]
        )

    else:

        st.success(
            results["status"]
        )

    # ---------------------------------
    # MAINTENANCE RECOMMENDATION
    # ---------------------------------

    st.subheader(
        "Maintenance Recommendation"
    )

    st.info(
        recommendation
    )

    # ---------------------------------
    # IMAGE PROCESSING RESULTS
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

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.hist(
        gray.ravel(),
        bins=256
    )

    ax.set_xlabel(
        "Pixel Intensity"
    )

    ax.set_ylabel(
        "Frequency"
    )

    st.pyplot(fig)

    # ---------------------------------
    # RGB TEMPERATURE VARIATION CURVES
    # ---------------------------------

    st.subheader(
        "RGB Temperature Variation Curves"
    )

    # Split RGB channels
    red_channel = img_rgb[:, :, 0]

    green_channel = img_rgb[:, :, 1]

    blue_channel = img_rgb[:, :, 2]

    # Mean intensity curves
    red_mean = np.mean(
        red_channel,
        axis=0
    )

    green_mean = np.mean(
        green_channel,
        axis=0
    )

    blue_mean = np.mean(
        blue_channel,
        axis=0
    )

    # Create plots
    fig2, axes = plt.subplots(
        1,
        3,
        figsize=(15,4)
    )

    # RED
    axes[0].plot(
        red_mean
    )

    axes[0].set_title(
        "Red Mean"
    )

    axes[0].set_xlabel(
        "Position"
    )

    axes[0].set_ylabel(
        "Intensity"
    )

    axes[0].grid(True)

    # GREEN
    axes[1].plot(
        green_mean
    )

    axes[1].set_title(
        "Green Mean"
    )

    axes[1].set_xlabel(
        "Position"
    )

    axes[1].set_ylabel(
        "Intensity"
    )

    axes[1].grid(True)

    # BLUE
    axes[2].plot(
        blue_mean
    )

    axes[2].set_title(
        "Blue Mean"
    )

    axes[2].set_xlabel(
        "Position"
    )

    axes[2].set_ylabel(
        "Intensity"
    )

    axes[2].grid(True)

    st.pyplot(
        fig2
    )

    # ---------------------------------
    # HOTSPOT ANALYSIS
    # ---------------------------------

    st.subheader(
        "Hotspot Analysis"
    )

    st.write(
        f"Hotspot Area Percentage: "
        f"{results['hotspot_percentage']:.2f}%"
    )

    st.success(
        "Analysis Completed Successfully"
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
            "Timestamp",
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

    h1.write("Timestamp")
    h2.write("Image")
    h3.write("Prediction")
    h4.write("Confidence")
    h5.write("Severity")
    h6.write("Delete")

    st.write("---")

    # ---------------------------------
    # SHOW HISTORY ROWS
    # ---------------------------------

    for index, row in history_data.iterrows():

        c1, c2, c3, c4, c5, c6 = st.columns(
            [2,2,2,2,2,1]
        )

        c1.write(
            row["Timestamp"]
        )

        c2.write(
            row["Image"]
        )

        c3.write(
            row["Prediction"]
        )

        c4.write(
            row["Confidence"]
        )

        c5.write(
            row["Severity"]
        )

        # DELETE BUTTON

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
