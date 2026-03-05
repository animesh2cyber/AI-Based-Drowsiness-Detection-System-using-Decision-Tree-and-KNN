import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import winsound

st.title("AI-Based Drowsiness Detection System")
st.write("Real-time Eye Monitoring using Decision Tree and KNN")

# ---------------- AI TRAINING ----------------

X_train = np.array([
    [0.35], [0.32], [0.30], [0.33], [0.34],
    [0.20], [0.18], [0.15], [0.22], [0.19]
])

y_train = np.array([1,1,1,1,1,0,0,0,0,0])

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

st.success("Models Trained Successfully!")

# ------------- EAR FUNCTION -----------------

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ------------- DLIB SETUP -----------------

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# ------------- START BUTTON -----------------

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
COUNTER = 0
ALARM_THRESHOLD = 15

if run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

            left_eye = [points[i] for i in LEFT_EYE]
            right_eye = [points[i] for i in RIGHT_EYE]

            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            prediction_dt = dt_model.predict([[ear]])[0]
            prediction_knn = knn_model.predict([[ear]])[0]

            if prediction_dt == 0 and prediction_knn == 0:
                COUNTER += 1
                if COUNTER >= ALARM_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS ALERT!",
                                (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)
                    winsound.Beep(2000, 1000)
                    st.error("DROWSINESS DETECTED!")
            else:
                COUNTER = 0

            cv2.putText(frame, f"EAR: {round(ear,2)}",
                        (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()