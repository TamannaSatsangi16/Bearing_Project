import cv2
import numpy as np
import joblib

# ---------------------------------
# LOAD MODEL
# ---------------------------------

model = joblib.load(
    "bearing_model.pkl"
)

# ---------------------------------
# LABELS
# ---------------------------------

labels = {

    0: "Healthy",

    1: "Old Used Bearing",

    2: "Iron Scrap Bearing",

    3: "Ball Defect Bearing"
}

# ---------------------------------
# PREDICTION
# ---------------------------------

def predict_bearing(img):

    # Resize
    img = cv2.resize(
        img,
        (128,128)
    )

    # Convert grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # Edge detection
    edges = cv2.Canny(
        gray,
        100,
        200
    )

    # Flatten
    feature = edges.flatten()

    feature = np.array([feature])

    # Prediction
    prediction = model.predict(feature)[0]

    # Confidence
    confidence = np.max(
        model.predict_proba(feature)
    ) * 100

    return (
        labels[prediction],
        round(confidence,2)
    )