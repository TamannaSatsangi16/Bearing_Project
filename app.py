import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import cv2

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

    # Open image
    image = Image.open(uploaded_file)

    # Convert to numpy array
    img = np.array(image)

    # Create uploads folder
    os.makedirs(
        "uploads",
        exist_ok=True
    )

    # Save uploaded image
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

    # Hotspot detection
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
    # RECOMMENDATION
    # ---------------------------------

    recommendation = get_recommendation(
        results["status"]
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

    history_df = pd.DataFrame([history])

    if os.path.exists("history.csv"):

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
        bins=256,
        color='gray'
    )
    
    ax.set_xlabel(
        "Pixel Intensity"
    )
    
    ax.set_ylabel(
        "Frequency"
    )
    
    st.pyplot(fig)
    
    # ---------------------------------
    # RGB CURVE
    # ---------------------------------
    
    st.subheader(
        "RGB Intensity Curve"
    )
    
    fig_rgb, ax_rgb = plt.subplots(
        figsize=(8,4)
    )
    
    colors = ('red', 'green', 'blue')
    
    for i, color in enumerate(colors):
    
        hist = cv2.calcHist(
            [img_rgb],
            [i],
            None,
            [256],
            [0,256]
        )
    
        ax_rgb.plot(
            hist,
            color=color,
            label=f'{color.upper()} Channel'
        )
    
    ax_rgb.set_xlim([0,256])
    
    ax_rgb.set_xlabel(
        "Pixel Intensity"
    )
    
    ax_rgb.set_ylabel(
        "Frequency"
    )
    
    ax_rgb.legend()
    
    st.pyplot(fig_rgb)
    
    # ---------------------------------
    # RGB MEANING
    # ---------------------------------
    
    st.markdown("""
    
    ### RGB Curve Meaning
    
    🔴 **Red Curve**
    - Represents hot regions
    - Higher red peak → higher temperature zones
    - Indicates overheating areas
    
    🟢 **Green Curve**
    - Represents medium temperature regions
    - Transitional thermal areas
    - Moderate heating indication
    
    🔵 **Blue Curve**
    - Represents cooler regions
    - Lower temperature areas
    - Normal/cool bearing surface
    
    """)
    
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

    # ---------------------------------
    # ANALYSIS HISTORY
    # ---------------------------------
    
    st.subheader(
        "Analysis History"
    )
    
    # Load history
    history_data = pd.read_csv(
        "history.csv"
    )
    
    # Show dataframe
    st.dataframe(
        history_data,
        use_container_width=True
    )

    # ---------------------------------
    # DELETE SINGLE RECORD
    # ---------------------------------
    
    st.subheader(
        "Delete Single Record"
    )
    
    delete_index = st.number_input(
    
        "Enter Row Number to Delete",
    
        min_value=0,
    
        max_value=max(
            len(history_data)-1,
            0
        ),
    
        step=1
    )
    
    if st.button(
        "Delete Selected Record"
    ):
    
        history_data = history_data.drop(
            delete_index
        )
    
        history_data = history_data.reset_index(
            drop=True
        )
    
        history_data.to_csv(
            "history.csv",
            index=False
        )
    
        st.success(
            "Record Deleted Successfully"
        )
    
        st.rerun()
    
    # ---------------------------------
    # DELETE ALL HISTORY
    # ---------------------------------
    
    st.subheader(
        "Delete Entire History"
    )
    
    if st.button(
        "Delete All Records"
    ):
    
        # Empty dataframe
        empty_df = pd.DataFrame(columns=[
            "Timestamp",
            "Image",
            "Prediction",
            "Confidence",
            "Severity",
            "Status"
        ])
    
        # Save empty CSV
        empty_df.to_csv(
            "history.csv",
            index=False
        )
    
        st.success(
            "All History Deleted Successfully"
        )
    
        st.rerun()
        # ---------------------------------
        # FINAL SUCCESS MESSAGE
        # ---------------------------------
    
        st.success(
            "Analysis Completed Successfully"
        )
