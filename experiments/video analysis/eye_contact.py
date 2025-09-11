import cv2
import mediapipe as mp
import numpy as np

# Mediapipe setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.9, min_tracking_confidence=0.9)

# Eye landmark sets
LEFT_EYE = {
    "left_corner": 130,
    "right_corner": 133,
    "top": 27,
    "bottom": 145
}
RIGHT_EYE = {
    "left_corner": 362,
    "right_corner": 263,
    "top": 257,
    "bottom": 374
}

# Iris centers
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Variable for multiplication with radius for dynamic height adjustment
# This needs to change for different screen sizes
# For 14-inch laptop, 0.15 works best. For 16-inch, 0.2 works best. For larger size, a greater multiplier must be used
RADIUS_MULTIPLIER = 0.2

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        def get_eye_center_and_radius(points):
            top_y = landmarks[points["top"]].y * h
            bottom_y = landmarks[points["bottom"]].y * h
            eye_height = bottom_y - top_y

            radius = eye_height * RADIUS_MULTIPLIER

            alpha = 0.65  # weighted center (flexible)
            y = int((1 - alpha) * top_y + alpha * bottom_y)
            x = int(((landmarks[points["left_corner"]].x + landmarks[points["right_corner"]].x) / 2) * w)
            return (x, y), radius

        def is_inside_circle(center, point, radius):
            dx = center[0] - point[0]
            dy = center[1] - point[1]
            distance = np.sqrt(dx**2 + dy**2)
            return distance <= radius

        # Left and right eye centers + dynamic radii
        left_eye_center, left_radius = get_eye_center_and_radius(LEFT_EYE)
        right_eye_center, right_radius = get_eye_center_and_radius(RIGHT_EYE)

        # Iris centers
        left_iris = (int(landmarks[LEFT_IRIS_CENTER].x * w), int(landmarks[LEFT_IRIS_CENTER].y * h))
        right_iris = (int(landmarks[RIGHT_IRIS_CENTER].x * w), int(landmarks[RIGHT_IRIS_CENTER].y * h))

        # Draw red eye center stickers
        cv2.circle(frame, left_eye_center, int(left_radius), (0, 0, 255), 2)
        cv2.circle(frame, right_eye_center, int(right_radius), (0, 0, 255), 2)

        # Draw green iris centers
        cv2.circle(frame, left_iris, 2, (0, 255, 0), -1)
        cv2.circle(frame, right_iris, 2, (0, 255, 0), -1)

        # Eye contact detection
        left_eye_contact = is_inside_circle(left_eye_center, left_iris, left_radius)
        right_eye_contact = is_inside_circle(right_eye_center, right_iris, right_radius)

        if left_eye_contact and right_eye_contact:
            text = "Eye Contact"
            color = (0, 255, 0)
        else:
            text = "No Eye Contact"
            color = (0, 0, 255)

        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Stable Eye Stickers + Iris + Eye Contact", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()