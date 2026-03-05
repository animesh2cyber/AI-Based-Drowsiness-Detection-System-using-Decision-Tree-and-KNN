import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import winsound

# -----------------------------
# AI TRAINING SECTION (SIMULATION)
# -----------------------------

# Sample training data (EAR values)
# 1 = Awake, 0 = Drowsy
X_train = np.array([
    [0.35], [0.32], [0.30], [0.33], [0.34],   # Awake
    [0.20], [0.18], [0.15], [0.22], [0.19]    # Drowsy
])

y_train = np.array([
    1,1,1,1,1,
    0,0,0,0,0
])

# Train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Train KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

print("Models Trained Successfully!")

# -----------------------------
# FUNCTION TO CALCULATE EAR
# -----------------------------

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# -----------------------------
# DLIB FACE + LANDMARK SETUP
# -----------------------------

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indexes
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Thresholds
COUNTER = 0
ALARM_THRESHOLD = 15  # frames

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Starting Drowsiness Detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = []

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

        left_eye = [landmarks_points[i] for i in LEFT_EYE]
        right_eye = [landmarks_points[i] for i in RIGHT_EYE]

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)

        ear = (leftEAR + rightEAR) / 2.0

        # AI Prediction
        prediction_dt = dt_model.predict([[ear]])[0]
        prediction_knn = knn_model.predict([[ear]])[0]

        # If both models predict drowsy
        if prediction_dt == 0 and prediction_knn == 0:
            COUNTER += 1

            if COUNTER >= ALARM_THRESHOLD:
                cv2.putText(frame, "DROWSINESS DETECTED!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)

                print("DROWSINESS DETECTED!")
                print("Sending Emergency Alert...")

                winsound.Beep(2000, 1000)

        else:
            COUNTER = 0

        # Draw eye contours
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"EAR: {round(ear,2)}",
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

    cv2.imshow("AI Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()