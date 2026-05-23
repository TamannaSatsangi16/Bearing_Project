import cv2
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ---------------------------------
# DATASET CLASSES
# ---------------------------------

classes = {

    "healthy": 0,

    "old_used": 1,

    "iron_scrap": 2,

    "ball_defect": 3
}

# ---------------------------------
# STORAGE
# ---------------------------------

X = []
y = []

# ---------------------------------
# LOAD IMAGES
# ---------------------------------

def load_images(folder, label):

    if not os.path.exists(folder):

        print(f"Folder not found: {folder}")
        return

    files = os.listdir(folder)

    print(f"\nLoading from: {folder}")

    for file in files:

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            path = os.path.join(
                folder,
                file
            )

            print(f"Reading: {path}")

            img = cv2.imread(path)

            if img is None:

                print(f"Could not read: {path}")
                continue

            # Resize
            img = cv2.resize(
                img,
                (128,128)
            )

            # Grayscale
            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            # Edge detection
            edges = cv2.Canny(
                gray,
                100,
                200
            )

            # Flatten
            feature = edges.flatten()

            X.append(feature)

            y.append(label)

# ---------------------------------
# LOAD ALL CLASSES
# ---------------------------------

for class_name, label in classes.items():

    folder_path = os.path.join(
        "dataset",
        class_name
    )

    load_images(
        folder_path,
        label
    )

# ---------------------------------
# NUMPY ARRAYS
# ---------------------------------

X = np.array(X)
y = np.array(y)

# ---------------------------------
# SPLIT DATA
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# MODEL
# ---------------------------------

model = SVC(
    kernel='linear',
    probability=True
)

# ---------------------------------
# TRAIN
# ---------------------------------

model.fit(
    X_train,
    y_train
)

# ---------------------------------
# TEST
# ---------------------------------

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nAccuracy: {accuracy}"
)

# ---------------------------------
# SAVE MODEL
# ---------------------------------

joblib.dump(
    model,
    "bearing_model.pkl"
)

print(
    "\nMULTICLASS MODEL TRAINED SUCCESSFULLY"
)