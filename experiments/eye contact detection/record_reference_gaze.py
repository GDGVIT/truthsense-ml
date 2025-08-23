import cv2
import mediapipe as mp
import numpy as np
import time

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

REFERENCE_VECTORS = []
SAVE_PATH = "reference_gaze_vectors.npy"

LEFT_EYE = [33, 133]
RIGHT_EYE = [263, 362]
LEFT_IRIS = 468
RIGHT_IRIS = 473

def get_normalized_eye_vector(landmarks, w, h):
    def get_vector(corner_idx, iris_idx):
        corner = np.array([
            landmarks[corner_idx].x * w,
            landmarks[corner_idx].y * h,
            landmarks[corner_idx].z * w  # approximate depth scale
        ])
        iris = np.array([
            landmarks[iris_idx].x * w,
            landmarks[iris_idx].y * h,
            landmarks[iris_idx].z * w
        ])
        vector = iris - corner
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else np.zeros(3)

    left_vec = get_vector(LEFT_EYE[0], LEFT_IRIS)
    right_vec = get_vector(RIGHT_EYE[0], RIGHT_IRIS)
    return (left_vec + right_vec) / 2  # average of both eyes

cap = cv2.VideoCapture(0)
print("Press 'c' to record a reference gaze vector (while looking into the camera).")
print("Press 's' to save and quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Draw eyes and iris
        for idx in LEFT_EYE + RIGHT_EYE + [LEFT_IRIS, RIGHT_IRIS]:
            cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            vec = get_normalized_eye_vector(landmarks, w, h)
            REFERENCE_VECTORS.append(vec)
            print(f"Captured vector: {vec}")
        elif key == ord('s'):
            break

    cv2.imshow("Recording Reference Gaze", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if REFERENCE_VECTORS:
    reference_array = np.array(REFERENCE_VECTORS)
    np.save(SAVE_PATH, reference_array)
    print(f"Saved {len(REFERENCE_VECTORS)} reference vectors to {SAVE_PATH}")
else:
    print("No vectors recorded.")