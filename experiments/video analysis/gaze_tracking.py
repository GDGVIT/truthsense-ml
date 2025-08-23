# Eye Gaze Estimation using OpenCV and the new MediaPipe FaceLandmarker Task
#
# This script uses the MediaPipe Tasks API to detect facial landmarks and
# estimates the direction of eye gaze based on the position of the irises
# relative to the eye corners.

import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

latest_result = None

def result_callback(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function to receive the detection result and store it.
    """
    global latest_result
    latest_result = result

base_options = python.BaseOptions(model_asset_path='../../posenet_models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False, # We don't need blendshapes for this task
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.LIVE_STREAM, # Use live stream mode for webcam
    result_callback=result_callback
)
landmarker = vision.FaceLandmarker.create_from_options(options)


# Define the indices for the left and right eye landmarks.
# These are specific to the MediaPipe Face Mesh model.
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

def get_landmark_point(landmarks, index, frame_shape):
    """
    Helper function to get the pixel coordinates of a landmark.
    """
    landmark_point = landmarks[index]
    x = int(landmark_point.x * frame_shape[1])
    y = int(landmark_point.y * frame_shape[0])
    return (x, y)

def get_gaze_direction(eye_points, iris_center):
    """
    Calculates the gaze direction based on the relative position of the iris.
    This version uses a ratio-based approach for more robust detection.
    Returns a string indicating the direction (e.g., "Up", "Down", "Left", "Right", "Center").
    """
    # Get horizontal and vertical points of the eye.
    # The indices are chosen based on the MediaPipe landmark map.
    left_corner = eye_points[0]   # Leftmost point (towards nose)
    right_corner = eye_points[8]  # Rightmost point (towards ear)
    top_corner = eye_points[12]   # Topmost point
    bottom_corner = eye_points[4] # Bottommost point

    # --- Horizontal Gaze Calculation (Ratio-based) ---
    eye_width = right_corner[0] - left_corner[0]
    if eye_width == 0: # Avoid division by zero
        horizontal_ratio = 0.5
    else:
        # Calculate the iris's horizontal position as a ratio of the eye's width
        horizontal_ratio = (iris_center[0] - left_corner[0]) / eye_width

    # --- Vertical Gaze Calculation (Ratio-based) ---
    eye_height = bottom_corner[1] - top_corner[1]
    if eye_height == 0: # Avoid division by zero
        vertical_ratio = 0.5
    else:
        # Calculate the iris's vertical position as a ratio of the eye's height
        vertical_ratio = (iris_center[1] - top_corner[1]) / eye_height

    # --- Determine Gaze Direction based on Ratios ---
    # These thresholds define the "center" zone and may need tuning for best results.
    horizontal_gaze = ""
    if horizontal_ratio > 0.65:
        horizontal_gaze = "Right"
    elif horizontal_ratio < 0.4:
        horizontal_gaze = "Left"

    vertical_gaze = ""
    if 0.35 < vertical_ratio < 0.45:
        vertical_gaze = "Up"
    elif vertical_ratio > 0.45: # Increased threshold for "Down" to avoid false positives
        vertical_gaze = "Down"

    # Combine gaze directions for a more descriptive output
    if vertical_gaze and horizontal_gaze:
        gaze_direction = f"{vertical_gaze}-{horizontal_gaze}"
    elif vertical_gaze:
        gaze_direction = vertical_gaze
    elif horizontal_gaze:
        gaze_direction = horizontal_gaze
    else:
        gaze_direction = "Center"

    return gaze_direction, horizontal_ratio, vertical_ratio


# Start capturing video from the webcam.
cap = cv2.VideoCapture(0)
frame_timestamp_ms = 0

print("Starting gaze estimation... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Get the current timestamp for the frame.
    frame_timestamp_ms = int(time.time() * 1000)

    # STEP 4: Detect face landmarks from the input image.
    landmarker.detect_async(mp_image, frame_timestamp_ms)

    # STEP 5: Process the detection result.
    if latest_result and latest_result.face_landmarks:
        for face_landmarks in latest_result.face_landmarks:
            landmarks = face_landmarks

            # Get the coordinates for the left and right eyes and irises
            left_eye_points = [get_landmark_point(landmarks, i, frame.shape) for i in LEFT_EYE_INDICES]
            right_eye_points = [get_landmark_point(landmarks, i, frame.shape) for i in RIGHT_EYE_INDICES]
            left_iris_points = [get_landmark_point(landmarks, i, frame.shape) for i in LEFT_IRIS_INDICES]
            right_iris_points = [get_landmark_point(landmarks, i, frame.shape) for i in RIGHT_IRIS_INDICES]

            # Calculate the center of the irises
            left_iris_center = (sum(p[0] for p in left_iris_points) // 4, sum(p[1] for p in left_iris_points) // 4)
            right_iris_center = (sum(p[0] for p in right_iris_points) // 4, sum(p[1] for p in right_iris_points) // 4)

            # Draw the eye and iris landmarks
            for point in left_eye_points:
                cv2.circle(frame, point, 1, (0, 255, 0), -1)
            for point in right_eye_points:
                cv2.circle(frame, point, 1, (0, 255, 0), -1)
            cv2.circle(frame, left_iris_center, 2, (0, 0, 255), -1)
            cv2.circle(frame, right_iris_center, 2, (0, 0, 255), -1)

            # Get the gaze direction for both eyes
            left_gaze, left_hr, left_vr = get_gaze_direction(left_eye_points, left_iris_center)
            right_gaze, right_hr, right_vr = get_gaze_direction(right_eye_points, right_iris_center)

            # Display the gaze direction on the screen
            cv2.putText(frame, f"Left Eye: {left_gaze} (Ratio: {left_hr:.4f}, {left_vr:.4f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Right Eye: {right_gaze} (Ratio: {right_hr:.4f}, {right_vr:.4f})", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Eye Gaze Estimation', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
landmarker.close()
cap.release()
cv2.destroyAllWindows()
