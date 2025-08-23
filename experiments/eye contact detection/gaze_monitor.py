import cv2
import mediapipe as mp
import numpy as np

# Load calibrated reference vectors
reference_vectors = np.load("reference_gaze_vectors.npy")
reference_vectors = np.array([v / np.linalg.norm(v) for v in reference_vectors])
reference_mean = np.mean(reference_vectors, axis=0)

# Setup MediaPipe
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Landmark indices
LEFT_EYE = [33, 133]
RIGHT_EYE = [263, 362]
LEFT_IRIS = 468
RIGHT_IRIS = 473

def get_gaze_vector(landmarks, w, h):
    def vector_between(corner_idx, iris_idx):
        corner = np.array([
            landmarks[corner_idx].x * w,
            landmarks[corner_idx].y * h,
            landmarks[corner_idx].z * w
        ])
        iris = np.array([
            landmarks[iris_idx].x * w,
            landmarks[iris_idx].y * h,
            landmarks[iris_idx].z * w
        ])
        vec = iris - corner
        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else np.zeros(3)

    left = vector_between(LEFT_EYE[0], LEFT_IRIS)
    right = vector_between(RIGHT_EYE[0], RIGHT_IRIS)
    return (left + right) / 2

# Setup webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

THRESHOLD_COSINE = 0.96  # Adjust if too strict/lenient

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get gaze direction vector
        current_vector = get_gaze_vector(landmarks, w, h)
        current_vector /= np.linalg.norm(current_vector)  # Normalize

        # Compute cosine similarity
        dot = np.dot(current_vector, reference_mean)
        dot = np.clip(dot, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))

        # 🧠 Debug sanity checks
        print("Current gaze vector:", current_vector)
        print("Reference mean vector:", reference_mean)
        print("Dot product:", dot)
        print("Angle (degrees):", angle_deg)

        # Feedback
        if dot > THRESHOLD_COSINE:
            status = f"Eye Contact ({angle_deg:.1f}°)"
            color = (0, 255, 0)
        else:
            status = f"Looking Away ({angle_deg:.1f}°)"
            color = (0, 0, 255)

        # Draw feedback
        cv2.putText(frame, status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Optional: draw gaze vector
        iris_center = np.array([
            (landmarks[LEFT_IRIS].x + landmarks[RIGHT_IRIS].x) * w / 2,
            (landmarks[LEFT_IRIS].y + landmarks[RIGHT_IRIS].y) * h / 2
        ]).astype(int)

        endpoint = iris_center + (current_vector[:2] * 80).astype(int)
        cv2.arrowedLine(frame, tuple(iris_center), tuple(endpoint), color, 2)

    cv2.imshow("Eye Contact Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()