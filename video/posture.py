import os
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

class BodyLanguageCorrector:
    def __init__(self):
        # Initialize MediaPipe tasks API components
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        
        # --- Robust Path Construction ---
        # Get the directory of the current script (posture.py)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Go up one level from 'video' to 'truthsense-ml' (your project root)
        project_root_dir = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
        model_dir = os.path.join(project_root_dir, 'mediapipe_models')

        self.pose_model_path = os.path.join(model_dir, 'pose_landmarker.task')
        self.face_model_path = os.path.join(model_dir, 'face_landmarker.task')
        self.hand_model_path = os.path.join(model_dir, 'hand_landmarker.task')
        
        # Debugging paths
        print(f"Loading models from directory: {model_dir}")
        print(f"Pose model path: {self.pose_model_path}, exists: {os.path.exists(self.pose_model_path)}")
        print(f"Hand model path: {self.hand_model_path}, exists: {os.path.exists(self.hand_model_path)}")
        print(f"Face model path: {self.face_model_path}, exists: {os.path.exists(self.face_model_path)}")

        # Initialize pose detector
        pose_options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_detector = self.PoseLandmarker.create_from_options(pose_options)
        
        # Initialize hand landmarker
        hand_options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = self.HandLandmarker.create_from_options(hand_options)

        # Initialize face landmarker (for eye/head pose) - DEFINED LAST AS REQUESTED
        face_options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.face_model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = self.FaceLandmarker.create_from_options(face_options)
        # self.face_landmarker = mp.solutions.face_mesh.FaceMesh(
        #     static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        #     min_detection_confidence=0.9, min_tracking_confidence=0.9
        #     )
                
        # Analysis parameters
        self.analysis_interval = 0.5  # 2 times per second
        self.last_analysis_time = 0
        self.total_frames_analyzed = 0

        # Global/Instance-level variables for eye contact & head pose (auto-calibration)
        self._is_calibrated = False
        self._reference_mean = np.array([0.0, 0.0, -1.0]) # Default, will be updated on first frame

        # Blink Detection Constants
        self.EAR_THRESHOLD = 0.2  # Threshold for Eye Aspect Ratio to detect a blink
        self.LEFT_EYE_EAR_POINTS = {"top": 159, "bottom": 145, "left": 33, "right": 133}
        self.RIGHT_EYE_EAR_POINTS = {"top": 386, "bottom": 374, "left": 362, "right": 263}

        # Eye landmark sets for circle-based detection (using FaceLandmarker indices)
        self.LEFT_EYE_CIRCLE_POINTS = {"left_corner": 130, "right_corner": 133, "top": 223, "bottom": 23}
        self.RIGHT_EYE_CIRCLE_POINTS = {"left_corner": 362, "right_corner": 359, "top": 443, "bottom": 253}
        self.LEFT_IRIS_CENTER_IDX = 468
        self.RIGHT_IRIS_CENTER_IDX = 473

        # Landmark indices for gaze vector calculation - consistent
        self.LEFT_EYE_GAZE_POINTS = [33, 133] # Medial and Lateral corners
        self.RIGHT_EYE_GAZE_POINTS = [263, 362]

        # For a 16-inch laptop, the right value for the Radius multiplier is 0.2 and THRESHOLD COSINE is 0.9
        self._RADIUS_MULTIPLIER = 0.2
        self._THRESHOLD_COSINE = 0.9  # Cosine similarity threshold for gaze direction (adjust as needed)

        # Categorized Feedback Counters
        self.feedback_counts: Dict[str, Dict[str, int]] = {
            "eye_contact": {
                "Eye contact maintained": 0,
                "Eyes off-center": 0,
                "No face detected": 0
            },
            "head_pose": {
                "Head centered": 0,
                "Head not centered": 0,
                "No face detected": 0
            },
            "shoulder_alignment": {
                "Good shoulder alignment": 0,
                "Slight shoulder tilt": 0,
                "Shoulders are tilted": 0,
                "No pose detected": 0
            },
            "head_alignment_body": { # Head alignment relative to body (from pose)
                "Good head alignment (body)": 0,
                "Head slightly off-center (body)": 0,
                "Keep your head centered over shoulders (body)": 0,
                "No pose detected": 0
            },
            # Removed "spine_straightness" category
            "gesture_usage": {
                "Use hand gestures to enhance your speech": 0,
                "Good use of gestures - consider using both hands": 0,
                "Great use of both hands for gesturing": 0,
                "No hands detected": 0
            },
            "hand_position": { # Specific to each hand if applicable, aggregate here
                "Hands in visible zone": 0,
                "Lower your hand slightly": 0,
                "Raise your hand for better visibility": 0,
                "No hands detected": 0
            }
        }
        
        # Current feedback string for each category
        self.current_feedback_status: Dict[str, str] = {
            "eye_contact": "Initializing...",
            "head_pose": "Initializing...",
            "shoulder_alignment": "Initializing...",
            "head_alignment_body": "Initializing...",
            # Removed "spine_straightness" status
            "gesture_usage": "Initializing...",
            "hand_position": "Initializing..."
        }

    def _update_feedback_counts(self, category: str, subtype: str):
        if category in self.feedback_counts:
            # Initialize subtype if it doesn't exist
            self.feedback_counts[category][subtype] = self.feedback_counts[category].get(subtype, 0) + 1
        else:
            print(f"Warning: Unknown feedback category '{category}'. Adding it.")
            self.feedback_counts[category] = {subtype: 1}

    def _calculate_eye_aspect_ratio(self, landmarks: List[Any], eye_points: Dict[str, int]) -> float:
        """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
        # Get 2D landmark coordinates
        top_pt = np.array([landmarks[eye_points["top"]].x, landmarks[eye_points["top"]].y])
        bottom_pt = np.array([landmarks[eye_points["bottom"]].x, landmarks[eye_points["bottom"]].y])
        left_pt = np.array([landmarks[eye_points["left"]].x, landmarks[eye_points["left"]].y])
        right_pt = np.array([landmarks[eye_points["right"]].x, landmarks[eye_points["right"]].y])

        # Calculate Euclidean distances
        vertical_dist = np.linalg.norm(top_pt - bottom_pt).item()
        horizontal_dist = np.linalg.norm(left_pt - right_pt).item()

        # Avoid division by zero
        if horizontal_dist == 0:
            return 0.0

        # Compute and return the EAR
        ear = vertical_dist / horizontal_dist
        return ear

    def _get_gaze_vector_from_facelandmarks(self, landmarks, w, h):
        """Calculates the average gaze vector from both eyes using FaceLandmarker output."""
        def _vector_between(corner_idx, iris_idx_obj):
            corner = np.array([landmarks[corner_idx].x * w, landmarks[corner_idx].y * h, landmarks[corner_idx].z * w])
            iris = np.array([iris_idx_obj.x * w, iris_idx_obj.y * h, iris_idx_obj.z * w])
            vec = iris - corner
            norm = np.linalg.norm(vec)
            return vec / norm if norm != 0 else np.zeros(3)

        left_vec = _vector_between(self.LEFT_EYE_GAZE_POINTS[0], landmarks[self.LEFT_IRIS_CENTER_IDX])
        right_vec = _vector_between(self.RIGHT_EYE_GAZE_POINTS[0], landmarks[self.RIGHT_IRIS_CENTER_IDX])
        
        avg_vec = (left_vec + right_vec) / 2
        norm_avg_vec = np.linalg.norm(avg_vec)
        return avg_vec / norm_avg_vec if norm_avg_vec != 0 else np.zeros(3)

    def _get_eye_detection_data_from_facelandmarks(self, landmarks, h, w):
        """Helper for circle-based eye detection using FaceLandmarker output."""
        def _get_eye_info(points_indices):
            top_y = landmarks[points_indices["top"]].y * h
            bottom_y = landmarks[points_indices["bottom"]].y * h
            radius = (bottom_y - top_y) * self._RADIUS_MULTIPLIER
            y = int((1 - 0.65) * top_y + 0.65 * bottom_y) # alpha = 0.65
            x = int(((landmarks[points_indices["left_corner"]].x + landmarks[points_indices["right_corner"]].x) / 2) * w)
            return (x, y), radius

        def _is_inside_circle(center, point, radius):
            return np.linalg.norm(np.array(center) - np.array(point)) <= radius

        left_eye_center, left_radius = _get_eye_info(self.LEFT_EYE_CIRCLE_POINTS)
        right_eye_center, right_radius = _get_eye_info(self.RIGHT_EYE_CIRCLE_POINTS)

        left_iris_pixel = (int(landmarks[self.LEFT_IRIS_CENTER_IDX].x * w), int(landmarks[self.LEFT_IRIS_CENTER_IDX].y * h))
        right_iris_pixel = (int(landmarks[self.RIGHT_IRIS_CENTER_IDX].x * w), int(landmarks[self.RIGHT_IRIS_CENTER_IDX].y * h))

        left_iris_in_bound = _is_inside_circle(left_eye_center, left_iris_pixel, left_radius)
        right_iris_in_bound = _is_inside_circle(right_eye_center, right_iris_pixel, right_radius)
        
        return left_iris_in_bound, right_iris_in_bound, left_eye_center, left_radius, right_eye_center, right_radius, left_iris_pixel, right_iris_pixel

    def _analyze_eye_contact_and_head_pose_internal(self, frame: np.ndarray, timestamp: int) -> Dict[str, Any]:
        """
        Internal function for eye contact & head pose, now uses instance's face_landmarker.
        Manages instance's auto-calibration variables.
        Also returns visualization data.
        """
        h, w = frame.shape[:2]
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        face_landmarker_result = self.face_landmarker.detect_for_video(mp_image, timestamp)
        # face_landmarker_result = self.face_landmarker.process(rgb_image)

        iris_in_bounds = False
        gaze_vector_aligned = False
        looking_at_camera = False
        confidence = 0.0
        eye_feedback_subtype = "No face detected"
        head_feedback_subtype = "No face detected"

        # --- Visualization Data Initialization ---
        # These will be populated if face landmarks are detected
        viz_data: dict = {
            "left_eye_center": None, "left_radius": None, "left_iris_pixel": None,
            "right_eye_center": None, "right_radius": None, "right_iris_pixel": None,
            "iris_center_combined": None, "gaze_endpoint": None, "gaze_color": None
        }

        if face_landmarker_result.face_landmarks:
        # if face_landmarker_result.multi_face_landmarks:
            landmarks = face_landmarker_result.face_landmarks[0]
            # landmarks = face_landmarker_result.multi_face_landmarks[0].landmark

            current_gaze_vector = self._get_gaze_vector_from_facelandmarks(landmarks, w, h)

            if not self._is_calibrated:
                self._reference_mean = current_gaze_vector
                self._is_calibrated = True
                print(f"Auto-calibrated reference gaze on first frame: {self._reference_mean}")
                # For the very first calibrated frame, assume perfect contact and set initial viz data
                h_f, w_f = frame.shape[:2]
                iris_center_combined = np.array([
                    (landmarks[self.LEFT_IRIS_CENTER_IDX].x + landmarks[self.RIGHT_IRIS_CENTER_IDX].x) * w_f / 2,
                    (landmarks[self.LEFT_IRIS_CENTER_IDX].y + landmarks[self.RIGHT_IRIS_CENTER_IDX].y) * h_f / 2
                ]).astype(int)
                
                viz_data["iris_center_combined"] = tuple(iris_center_combined)
                viz_data["gaze_endpoint"] = tuple(iris_center_combined + (current_gaze_vector[:2] * 80).astype(int))
                viz_data["gaze_color"] = (0, 255, 0) # Green for good gaze on calibration frame

                return {
                    "looking_at_camera": True, "iris_in_bounds": True, "gaze_vector_aligned": True,
                    "confidence": 1.0, 
                    "eye_feedback_subtype": "Eye contact maintained", 
                    "head_feedback_subtype": "Head centered",
                    "viz_data": viz_data # Return viz data for drawing
                }

            left_iris_in_bound, right_iris_in_bound, left_eye_center, left_radius, \
                right_eye_center, right_radius, left_iris_pixel, right_iris_pixel = \
                self._get_eye_detection_data_from_facelandmarks(landmarks, h, w)
            iris_in_bounds = left_iris_in_bound and right_iris_in_bound

            # Check for blinking
            left_ear = self._calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_EAR_POINTS)
            right_ear = self._calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_EAR_POINTS)

            if left_ear < self.EAR_THRESHOLD and right_ear < self.EAR_THRESHOLD:
                # Both eyes are closed, it's a blink.
                return {
                    "looking_at_camera": False, "iris_in_bounds": False,
                    "gaze_vector_aligned": False, "confidence": 0.0,
                    "eye_feedback_subtype": "Blinking",
                    "head_feedback_subtype": "Head centered",  # Assume head pose is okay during a blink
                    "viz_data": viz_data # Return empty viz_data to prevent drawing
                }

            # Populate viz_data for drawing eye circles and iris points
            viz_data["left_eye_center"] = left_eye_center
            viz_data["left_radius"] = left_radius
            viz_data["left_iris_pixel"] = left_iris_pixel
            viz_data["right_eye_center"] = right_eye_center
            viz_data["right_radius"] = right_radius
            viz_data["right_iris_pixel"] = right_iris_pixel


            dot_product = np.dot(current_gaze_vector, self._reference_mean)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            gaze_vector_aligned = dot_product > self._THRESHOLD_COSINE

            looking_at_camera = iris_in_bounds and gaze_vector_aligned

            if looking_at_camera:
                confidence = 1.0
            elif iris_in_bounds or gaze_vector_aligned:
                confidence = 0.5
            else:
                confidence = 0.0

            confidence_gaze_component = (dot_product - self._THRESHOLD_COSINE) / (1.0 - self._THRESHOLD_COSINE) if (1.0 - self._THRESHOLD_COSINE) > 0 else 0
            confidence_gaze_component = np.clip(confidence_gaze_component, 0, 1)

            if iris_in_bounds:
                confidence = (confidence + confidence_gaze_component) / 2
            else:
                confidence = 0.0

            eye_feedback_subtype = "Eyes off-center" if not iris_in_bounds else "Eye contact maintained"
            head_feedback_subtype = "Head not centered" if not gaze_vector_aligned else "Head centered"

            # Populate viz_data for drawing gaze vector
            iris_center_combined = np.array([
                (landmarks[self.LEFT_IRIS_CENTER_IDX].x + landmarks[self.RIGHT_IRIS_CENTER_IDX].x) * w / 2,
                (landmarks[self.LEFT_IRIS_CENTER_IDX].y + landmarks[self.RIGHT_IRIS_CENTER_IDX].y) * h / 2
            ]).astype(int)
            viz_data["iris_center_combined"] = tuple(iris_center_combined)
            viz_data["gaze_endpoint"] = tuple(iris_center_combined + (current_gaze_vector[:2] * 80).astype(int))
            viz_data["gaze_color"] = (0, 255, 0) if gaze_vector_aligned else (0, 0, 255)

        return {
            "looking_at_camera": looking_at_camera,
            "iris_in_bounds": iris_in_bounds,
            "gaze_vector_aligned": gaze_vector_aligned,
            "confidence": float(confidence),
            "eye_feedback_subtype": eye_feedback_subtype,
            "head_feedback_subtype": head_feedback_subtype,
            "viz_data": viz_data # Always return viz_data, even if empty
        }

    def _analyze_posture(self, pose_landmarks) -> Dict[str, Any]:
        """Analyze posture for public speaking."""
        
        landmarks = pose_landmarks # pose_landmarks is already the list of landmarks
        
        # Shoulder alignment
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        
        shoulder_feedback = ""
        if shoulder_slope < 0.05:
            shoulder_feedback = "Good shoulder alignment"
        elif shoulder_slope < 0.1:
            shoulder_feedback = "Slight shoulder tilt"
        else:
            shoulder_feedback = "Shoulders are tilted"
        self._update_feedback_counts("shoulder_alignment", shoulder_feedback)
        self.current_feedback_status["shoulder_alignment"] = shoulder_feedback
        
        # Head position (relative to shoulders from pose landmarks)
        nose = landmarks[0]
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        
        head_alignment = abs(nose.x - shoulder_center_x)
        
        head_body_feedback = ""
        if head_alignment < 0.03:
            head_body_feedback = "Good head alignment (body)"
        elif head_alignment < 0.06:
            head_body_feedback = "Head slightly off-center (body)"
        else:
            head_body_feedback = "Keep your head centered over shoulders (body)"
        self._update_feedback_counts("head_alignment_body", head_body_feedback)
        self.current_feedback_status["head_alignment_body"] = head_body_feedback
        
        # Removed spine straightness analysis code
        
        return {
            "shoulder_alignment_feedback": shoulder_feedback,
            "head_alignment_body_feedback": head_body_feedback,
            # Removed "spine_straightness_feedback" from return
        }

    def _analyze_hand_gestures(self, hand_landmarks_list) -> Dict[str, Any]:
        """Analyze hand gestures for public speaking."""
        
        num_hands = len(hand_landmarks_list)
        
        gesture_usage_feedback = ""
        if num_hands == 0:
            gesture_usage_feedback = "Use hand gestures to enhance your speech"
        elif num_hands == 1:
            gesture_usage_feedback = "Good use of gestures - consider using both hands"
        else:
            gesture_usage_feedback = "Great use of both hands for gesturing"
        self._update_feedback_counts("gesture_usage", gesture_usage_feedback)
        self.current_feedback_status["gesture_usage"] = gesture_usage_feedback

        hand_position_feedback_list = []
        if num_hands > 0:
            for i, hand_landmarks in enumerate(hand_landmarks_list):
                wrist = hand_landmarks[0]
                
                hand_pos_feedback = "Hands in visible zone"
                if wrist.y < 0.3:  # Too high
                    hand_pos_feedback = "Lower your hand slightly"
                elif wrist.y > 0.8:  # Too low
                    hand_pos_feedback = "Raise your hand for better visibility"
                
                hand_position_feedback_list.append(hand_pos_feedback)
                self._update_feedback_counts("hand_position", hand_pos_feedback)
                
            # Aggregate hand position feedback into a single current status string
            # Check if any hand needs adjustment, otherwise assume good
            if any("Lower" in fb for fb in hand_position_feedback_list):
                self.current_feedback_status["hand_position"] = "Adjust hand height: Lower"
            elif any("Raise" in fb for fb in hand_position_feedback_list):
                self.current_feedback_status["hand_position"] = "Adjust hand height: Raise"
            else:
                self.current_feedback_status["hand_position"] = "Hands in visible zone"
        else:
            self._update_feedback_counts("hand_position", "No hands detected")
            self.current_feedback_status["hand_position"] = "No hands detected"


        return {
            "gesture_usage_feedback": gesture_usage_feedback,
            "hand_position_feedback_list": hand_position_feedback_list
        }

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> Dict[str, Any]:
        """
        Process a single frame and return comprehensive analysis results
        including categorized feedback and updates internal counters.
        """
        self.total_frames_analyzed += 1

        # Convert BGR to RGB for all MediaPipe detectors
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect pose
        pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        
        # Detect hands
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Analyze eye contact and head pose using the instance's FaceLandmarker (face_landmarker is initialized last)
        eye_head_result = self._analyze_eye_contact_and_head_pose_internal(frame, timestamp_ms)
        
        # Update eye contact feedback counts
        self._update_feedback_counts("eye_contact", eye_head_result["eye_feedback_subtype"])
        self.current_feedback_status["eye_contact"] = eye_head_result["eye_feedback_subtype"]
        
        # Update head pose feedback counts
        self._update_feedback_counts("head_pose", eye_head_result["head_feedback_subtype"])
        self.current_feedback_status["head_pose"] = eye_head_result["head_feedback_subtype"]

        # Initialize overall feedback and scores for this frame
        overall_feedback_list = []
        
        # Analyze posture if pose detected
        if pose_result.pose_landmarks:
            posture_analysis = self._analyze_posture(pose_result.pose_landmarks[0])
            overall_feedback_list.extend([
                posture_analysis["shoulder_alignment_feedback"],
                posture_analysis["head_alignment_body_feedback"],
                # Removed "spine_straightness_feedback"
            ])
        else:
            overall_feedback_list.append("No pose detected")
            self._update_feedback_counts("shoulder_alignment", "No pose detected")
            self._update_feedback_counts("head_alignment_body", "No pose detected")
            self.current_feedback_status["shoulder_alignment"] = "No pose detected"
            self.current_feedback_status["head_alignment_body"] = "No pose detected"

        # Analyze hand gestures if hands detected
        if hand_result.hand_landmarks:
            gesture_analysis = self._analyze_hand_gestures(hand_result.hand_landmarks)
            # hand_position_feedback_list is already handled for counts in _analyze_hand_gestures
            overall_feedback_list.append(gesture_analysis["gesture_usage_feedback"])
        else:
            overall_feedback_list.append("No hands detected")
            self._update_feedback_counts("gesture_usage", "No hands detected")
            self._update_feedback_counts("hand_position", "No hands detected")
            self.current_feedback_status["gesture_usage"] = "No hands detected"
            self.current_feedback_status["hand_position"] = "No hands detected"

        # Combine results into the final structure for this frame
        final_feedback_for_frame = {
            "eye_contact": {
                "feedback": self.current_feedback_status["eye_contact"],
                "confidence": eye_head_result["confidence"]
            },
            "head_pose": {
                "feedback": self.current_feedback_status["head_pose"],
                "gaze_aligned": eye_head_result["gaze_vector_aligned"]
            },
            "shoulder_alignment": {
                "feedback": self.current_feedback_status["shoulder_alignment"]
            },
            "head_alignment_body": {
                "feedback": self.current_feedback_status["head_alignment_body"]
            },
            # Removed "spine_straightness" from final feedback
            "gesture_usage": {
                "feedback": self.current_feedback_status["gesture_usage"]
            },
            "hand_position": {
                "feedback": self.current_feedback_status["hand_position"]
            },
            "overall_feedback_list": overall_feedback_list, # For consolidated display if needed
            "viz_data": eye_head_result["viz_data"] # Pass through visualization data
        }
        
        return final_feedback_for_frame

    def run_video_analysis(self, video_source=0):
        """
        Run real-time video analysis and print categorized feedback percentages at the end.
        video_source: 0 for webcam, or path to video file
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("\n--- Starting Real-time Body Language Analysis ---")
        print("Please look at the camera when the stream starts for auto-calibration.")
        print("Press 'q' to quit.")
        
        # Reset auto-calibration variables for each new run
        self._is_calibrated = False 
        self._reference_mean = np.array([0.0, 0.0, -1.0])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            timestamp_ms = int(current_time * 1000)
            
            # Process frame and get feedback results
            feedback_for_frame = self.process_frame(frame, timestamp_ms) 
            
            # --- VISUALIZATION CODE ---
            # This section draws all the feedback text and eye contact visuals on the frame
            viz_data = feedback_for_frame["viz_data"]
            h, w = frame.shape[:2]
            is_blinking = feedback_for_frame["eye_contact"]["feedback"] == "Blinking"

            # Only draw the eye circles and gaze vector if the user is NOT blinking
            # and the necessary visualization data exists.

            # 1. Visualize Eye Contact (Circles and Gaze Vector)

            if viz_data["left_eye_center"] is not None and not is_blinking:
                # Left Eye
                cv2.circle(frame, viz_data["left_eye_center"], int(viz_data["left_radius"]), 
                           (0, 0, 255) if not feedback_for_frame["eye_contact"]["feedback"] == "Eye contact maintained" else (0, 255, 0), 2)
                cv2.circle(frame, viz_data["left_iris_pixel"], 2, 
                           (0, 0, 255) if not feedback_for_frame["eye_contact"]["feedback"] == "Eye contact maintained" else (0, 255, 0), -1)

                # Right Eye
                cv2.circle(frame, viz_data["right_eye_center"], int(viz_data["right_radius"]), 
                           (0, 0, 255) if not feedback_for_frame["eye_contact"]["feedback"] == "Eye contact maintained" else (0, 255, 0), 2)
                cv2.circle(frame, viz_data["right_iris_pixel"], 2, 
                           (0, 0, 255) if not feedback_for_frame["eye_contact"]["feedback"] == "Eye contact maintained" else (0, 255, 0), -1)

                # Draw Gaze Vector
                if viz_data["iris_center_combined"] is not None and viz_data["gaze_endpoint"] is not None:
                    cv2.arrowedLine(frame, viz_data["iris_center_combined"], viz_data["gaze_endpoint"], viz_data["gaze_color"], 2)


            # 2. Visualize Text Feedback
            # Define text display parameters
            text_x_start = 30
            text_y_start = 50
            line_height = 20 # Changed to 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 # Adjusted for smaller line height
            text_color = (0, 0, 0) # BLACK (B, G, R)
            good_color = (0, 255, 0) # Green (Still for circle/arrow)
            bad_color = (0, 0, 255) # Red (Still for circle/arrow)
            warning_color = (0, 255, 255) # Yellow (Still for circle/arrow)

            # Display Status
            status_text = "Status: " + ("Looking at Camera" if feedback_for_frame["eye_contact"]["feedback"] == "Eye contact maintained" else "Needs Improvement")
            cv2.putText(frame, status_text, (text_x_start, text_y_start), font, font_scale, text_color, 2)

            # Display Confidence (for eye contact)
            confidence_text = f"Eye Confidence: {feedback_for_frame['eye_contact']['confidence']:.2f}"
            cv2.putText(frame, confidence_text, (text_x_start, text_y_start + line_height), font, font_scale, text_color, 2)
            
            # Display Eye Feedback
            eye_fb_text = f"Eye: {feedback_for_frame['eye_contact']['feedback']}"
            cv2.putText(frame, eye_fb_text, (text_x_start, text_y_start + 2 * line_height), font, font_scale, text_color, 2)
            
            # Display Head Pose Feedback
            head_fb_text = f"Head: {feedback_for_frame['head_pose']['feedback']}"
            cv2.putText(frame, head_fb_text, (text_x_start, text_y_start + 3 * line_height), font, font_scale, text_color, 2)

            # Display Shoulder Alignment Feedback
            shoulder_fb_text = f"Shoulders: {feedback_for_frame['shoulder_alignment']['feedback']}"
            cv2.putText(frame, shoulder_fb_text, (text_x_start, text_y_start + 4 * line_height), font, font_scale, text_color, 2)

            # Display Head Alignment (Body) Feedback
            head_body_fb_text = f"Head (Body): {feedback_for_frame['head_alignment_body']['feedback']}"
            cv2.putText(frame, head_body_fb_text, (text_x_start, text_y_start + 5 * line_height), font, font_scale, text_color, 2)

            # Removed Spine Straightness Feedback display
            # line_for_spine = 6 # If spine was present
            # cv2.putText(frame, spine_fb_text, (text_x_start, text_y_start + line_for_spine * line_height), font, font_scale, text_color, 2)

            # Display Gesture Usage Feedback (Adjust Y position due to removed spine line)
            gesture_fb_text = f"Gestures: {feedback_for_frame['gesture_usage']['feedback']}"
            cv2.putText(frame, gesture_fb_text, (text_x_start, text_y_start + 6 * line_height), font, font_scale, text_color, 2) # Now on line 6

            # Display Hand Position Feedback (Adjust Y position)
            hand_pos_fb_text = f"Hand Pos: {feedback_for_frame['hand_position']['feedback']}"
            cv2.putText(frame, hand_pos_fb_text, (text_x_start, text_y_start + 7 * line_height), font, font_scale, text_color, 2) # Now on line 7


            # Calibration prompt
            if not self._is_calibrated:
                cv2.putText(frame, "Auto-Calibrating: Please look at camera...", 
                            (frame.shape[1] // 2 - 250, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


            # Display the frame
            cv2.imshow('Body Language Corrector', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # --- Print Final Percentage Ratios ---
        self._print_final_analysis_report()

    def _print_final_analysis_report(self):
        print("\n" + "="*50)
        print(f"Body Language Analysis Report ({self.total_frames_analyzed} frames analyzed)")
        print("="*50)

        if self.total_frames_analyzed == 0:
            print("No frames were analyzed during this session.")
            return

        for category, subtypes in self.feedback_counts.items():
            print(f"\n--- {category.replace('_', ' ').title()} ---")
            total_category_count = sum(subtypes.values())
            
            # Handle cases where a category might have zero count for all its subtypes
            if total_category_count == 0:
                print(f"  (No data for {category.replace('_', ' ')} during this session)")
                continue

            for subtype, count in subtypes.items():
                # Percentage based on total frames analyzed, not total_category_count
                percentage = (count / self.total_frames_analyzed) * 100 
                print(f"  - {subtype}: {percentage:.2f}% ({count} frames)")
        
        print("\n" + "="*50)
        print("Analysis Complete.")
        print("="*50)


# Usage example
if __name__ == "__main__":
    corrector = BodyLanguageCorrector()
    corrector.run_video_analysis(0) # Use webcam (0) or path to video file