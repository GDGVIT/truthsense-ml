import os
import cv2
import json
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions, 
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    RunningMode
)
from typing import Dict, List, Tuple, Any, Optional

class BodyLanguageCorrector:
    def __init__(self):        
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../posenet_models')
        self.pose_model_path = os.path.join(model_dir, 'pose_landmarker.task')
        self.face_model_path = os.path.join(model_dir, 'face_landmarker.task')
        self.hand_model_path = os.path.join(model_dir, 'hand_landmarker.task')
        
        # Debugging paths
        print(f"Loading models from directory: {model_dir}")
        print(f"Pose model path: {self.pose_model_path}, exists: {os.path.exists(self.pose_model_path)}")
        print(f"Hand model path: {self.hand_model_path}, exists: {os.path.exists(self.hand_model_path)}")
        print(f"Face model path: {self.face_model_path}, exists: {os.path.exists(self.face_model_path)}")

        # Initialize pose detector
        self.pose_detector = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.pose_model_path),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )
        
        # Initialize hand landmarker
        self.hand_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )

        # Initialize face landmarker
        self.face_landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.face_model_path),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
        )
                
        # Analysis parameters
        self.analysis_interval = 0.5  # 2 Hertz frequency
        self.last_analysis_time = 0
        self.total_frames = 0

        # Global/Instance-level variables for eye contact & head pose calibration
        self._is_calibrated = False
        self._reference_mean = np.array([0.0, 0.0, -1.0]) # Default, will be updated on first frame

        # Blink Detection Constants
        self.EAR_THRESHOLD = 0.2  # Threshold for Eye Aspect Ratio to detect a blink
        self.LEFT_EYE_EAR_POINTS = {"top": 159, "bottom": 145, "left": 33, "right": 133}
        self.RIGHT_EYE_EAR_POINTS = {"top": 386, "bottom": 374, "left": 362, "right": 263}

        # A counter for eye contact proxied confidence estimation
        self.eye_contact_confidence = {"sum": 0, "count": 0}

        # Eye landmark sets for circle-based detection (using FaceLandmarker indices)
        self.LEFT_EYE_CIRCLE_POINTS = {"left_corner": 130, "right_corner": 133, "top": 223, "bottom": 23}
        self.RIGHT_EYE_CIRCLE_POINTS = {"left_corner": 362, "right_corner": 359, "top": 443, "bottom": 253}
        self.LEFT_IRIS_CENTER_IDX = 468
        self.RIGHT_IRIS_CENTER_IDX = 473

        # Landmark indices for gaze vector calculation - consistent
        self.LEFT_EYE_GAZE_POINTS = [33, 133] # Medial and Lateral corners
        self.RIGHT_EYE_GAZE_POINTS = [263, 362]

        # For a 16-inch laptop, the right value for the Radius multiplier is 0.2 and THRESHOLD COSINE is 0.9
        # For a 14-inch laptop, radius multiplier should be 0.15
        self.RADIUS_MULTIPLIER = 0.2
        self.THRESHOLD_COSINE = 0.9

        # Initialize feedback counters
        self.feedback_counts: Dict[str, Dict[str, int]] = {
            "eyeContact": {},
            "shoulderAlignment": {},
            "headBodyAlignment": {},
            "hands": {}
        }
    
    def _calculate_EAR(self, landmarks: List[Any], points: Dict[str, int]) -> float:
        """Calculate eye aspect ratio to detect blinks"""
        def calculate_distance(a, b):
            return np.linalg.norm([a.x - b.x, a.y - b.y])

        top = landmarks[points["top"]]
        bottom = landmarks[points["bottom"]]
        left = landmarks[points["left"]]
        right = landmarks[points["right"]]

        vertical_dist = calculate_distance(top, bottom) or 0.0
        horizontal_dist = calculate_distance(left, right) or 0.0

        return vertical_dist / horizontal_dist if horizontal_dist != 0 else 0.0     # type: ignore

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
        """Helper function for circle-based eye detection using FaceLandmarker output."""
        
        def _get_eye_info(points_indices, alpha=0.65):
            top_y = landmarks[points_indices["top"]].y * h
            bottom_y = landmarks[points_indices["bottom"]].y * h
            radius = (bottom_y - top_y) * self.RADIUS_MULTIPLIER
            y = int((1 - alpha) * top_y + alpha * bottom_y)
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

    def eye_and_head_analysis(self, frame: np.ndarray, timestamp: int) -> Dict[str, Any]:
        """
        Function that performs analysis of head pose and eye contact.
        Can also return visualization data.
        """
        h, w = frame.shape[:2]
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        face_landmarker_result = self.face_landmarker.detect_for_video(mp_image, timestamp)

        iris_in_bounds = False
        gaze_vector_aligned = False
        looking_at_camera = False
        confidence = 0.0
        eye_feedback_subtype = "No face detected"
        head_feedback_subtype = "No face detected"

        viz_data: dict = {
            "left_eye_center": None, "left_radius": None, "left_iris_pixel": None,
            "right_eye_center": None, "right_radius": None, "right_iris_pixel": None,
            "iris_center_combined": None, "gaze_endpoint": None, "gaze_color": None
        }

        if face_landmarker_result.face_landmarks:
            landmarks = face_landmarker_result.face_landmarks[0]

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
                    "viz_data": viz_data
                }

            left_iris_in_bound, right_iris_in_bound, left_eye_center, left_radius, \
                right_eye_center, right_radius, left_iris_pixel, right_iris_pixel = \
                self._get_eye_detection_data_from_facelandmarks(landmarks, h, w)
            iris_in_bounds = left_iris_in_bound and right_iris_in_bound

            viz_data["left_eye_center"] = left_eye_center
            viz_data["left_radius"] = left_radius
            viz_data["left_iris_pixel"] = left_iris_pixel
            viz_data["right_eye_center"] = right_eye_center
            viz_data["right_radius"] = right_radius
            viz_data["right_iris_pixel"] = right_iris_pixel

            # Check for blinking
            left_ear = self._calculate_EAR(landmarks, self.LEFT_EYE_EAR_POINTS)
            right_ear = self._calculate_EAR(landmarks, self.RIGHT_EYE_EAR_POINTS)

            if left_ear < self.EAR_THRESHOLD and right_ear < self.EAR_THRESHOLD:
                # Both eyes are closed, it's a blink.
                return {
                    "looking_at_camera": False, "iris_in_bounds": False,
                    "gaze_vector_aligned": False, "confidence": 0.0,
                    "eye_feedback_subtype": "Blinking",
                    "head_feedback_subtype": "Head centered",
                    "viz_data": viz_data # Return empty viz_data to prevent drawing
                }

            # Check whether the user is looking at the screen
            dot_product = np.dot(current_gaze_vector, self._reference_mean)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            gaze_vector_aligned = dot_product > self.THRESHOLD_COSINE

            looking_at_camera = iris_in_bounds and gaze_vector_aligned

            if looking_at_camera:
                confidence = 1.0
            elif iris_in_bounds or gaze_vector_aligned:
                confidence = 0.5
            else:
                confidence = 0.0

            confidence_gaze_component = (dot_product - self.THRESHOLD_COSINE) / (1.0 - self.THRESHOLD_COSINE) if (1.0 - self.THRESHOLD_COSINE) > 0 else 0
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
            "eye_feedback_subtype": eye_feedback_subtype + "; " + head_feedback_subtype,
            "viz_data": viz_data # Always return viz_data, even if empty
        }

    def posture_analysis(self, landmarks) -> Dict[str, Any]:
        """Analyze posture for public speaking."""
        
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
        
        # Head position (relative to shoulders)
        nose = landmarks[0]
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        
        head_alignment = abs(nose.x - shoulder_center_x)
        
        head_alignment_feedback = ""
        if head_alignment < 0.03:
            head_alignment_feedback = "Head aligned with body"
        elif head_alignment < 0.06:
            head_alignment_feedback = "Head slightly tilted"
        else:
            head_alignment_feedback = "Head tilted"

        # Update feedback counts
        self.update_feedback("headBodyAlignment", head_alignment_feedback)
        self.update_feedback("shoulderAlignment", shoulder_feedback)

        return {
            "shoulderAlignment": shoulder_feedback,
            "headAlignment": head_alignment_feedback,
        }

    def gesture_analysis(self, hand_landmarks_list) -> str:
        """Analyze and detect hand placement"""

        gestures_feedback = ""

        if (not hand_landmarks_list): gestures_feedback += "Hands not in frame"
        elif (len(hand_landmarks_list) == 1):
            gestures_feedback += "One hand in frame; "
            wrist = hand_landmarks_list[0][0]
            if (wrist.y < 0.5):
                gestures_feedback += "Hand high in the frame"
            elif (0.5 < wrist.y < 0.9):
                gestures_feedback += "Hand in the middle of the frame"
            else:
                gestures_feedback += "Hand low in the frame"
        
        else:
            gestures_feedback += "Both hands in frame; "    
            for i in range(2):
                wrist = hand_landmarks_list[i][0]
                if (wrist.y < 0.5):
                    gestures_feedback += "One hand high in the frame"
                elif (0.5 < wrist.y < 0.9):
                    gestures_feedback += "One hand in the middle of the frame"
                else:
                    gestures_feedback += "One hand low in the frame"
                
                if (i == 0): gestures_feedback += "; "

        self.update_feedback('hands', gestures_feedback)
        return gestures_feedback

    def update_feedback(self, category: str, subtype: str):
        if subtype not in self.feedback_counts[category]:
            self.feedback_counts[category][subtype] = 1
        else:
            self.feedback_counts[category][subtype] += 1

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> Dict[str, Any]:
        """
        Process a single frame and return comprehensive analysis results.
        Matches the TypeScript structure from the frontend.
        """
        self.total_frames += 1

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        eye_head_result = self.eye_and_head_analysis(frame, timestamp_ms)
        
        # Update eye contact confidence tracking
        self.eye_contact_confidence["sum"] += eye_head_result["confidence"]
        avg_confidence = self.eye_contact_confidence["sum"] / self.total_frames
        
        # Update feedback counts for eye contact
        self.update_feedback("eyeContact", eye_head_result["eye_feedback_subtype"])
        
        # Initialize feedback structure
        feedback = {
            "eyeContact": {
                eye_head_result["eye_feedback_subtype"]: 1,
                "confidence": avg_confidence
            },
            "shoulderAlignment": {},
            "headBodyAlignment": {},
            "hands": {}
        }

        # Analyze posture if pose detected
        if pose_result.pose_landmarks:
            posture_analysis = self.posture_analysis(pose_result.pose_landmarks[0])
            feedback["shoulderAlignment"][posture_analysis["shoulderAlignment"]] = 1
            feedback["headBodyAlignment"][posture_analysis["headAlignment"]] = 1
        else:
            feedback["shoulderAlignment"]["No pose detected"] = 1
            feedback["headBodyAlignment"]["No pose detected"] = 1

        # Analyze hand gestures if hands detected
        gesture_feedback = self.gesture_analysis(hand_result.hand_landmarks)
        feedback["hands"][gesture_feedback] = 1
        
        return {
            "feedback": feedback,
            "viz_data": eye_head_result["viz_data"]
        }

    def run_video_analysis(self, video_source=0, show_viz=True, display=True, save_location=""):
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
            
            if show_viz:
                viz_data = feedback_for_frame["viz_data"]
                is_blinking = feedback_for_frame["feedback"]["eyeContact"] == "Blinking"

                if viz_data["left_eye_center"] is not None and not is_blinking:
                    # Left Eye
                    cv2.circle(frame, viz_data["left_eye_center"], int(viz_data["left_radius"]), 
                            (0, 0, 255) if not feedback_for_frame["feedback"]["eyeContact"] == "Eye contact maintained" else (0, 255, 0), 2)
                    cv2.circle(frame, viz_data["left_iris_pixel"], 2, 
                            (0, 0, 255) if not feedback_for_frame["feedback"]["eyeContact"] == "Eye contact maintained" else (0, 255, 0), -1)

                    # Right Eye
                    cv2.circle(frame, viz_data["right_eye_center"], int(viz_data["right_radius"]), 
                            (0, 0, 255) if not feedback_for_frame["feedback"]["eyeContact"] == "Eye contact maintained" else (0, 255, 0), 2)
                    cv2.circle(frame, viz_data["right_iris_pixel"], 2, 
                            (0, 0, 255) if not feedback_for_frame["feedback"]["eyeContact"] == "Eye contact maintained" else (0, 255, 0), -1)

                    # Draw Gaze Vector
                    if viz_data["iris_center_combined"] is not None and viz_data["gaze_endpoint"] is not None:
                        cv2.arrowedLine(frame, viz_data["iris_center_combined"], viz_data["gaze_endpoint"], viz_data["gaze_color"], 2)


                # Define text display parameters
                text_x_start = 30
                text_y_start = 50
                line_height = 20 
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6 
                text_color = (0, 0, 0) # BLACK

                # Display Confidence (for eye contact)
                confidence_text = f"Eye Confidence: {feedback_for_frame['feedback']['eyeContact']['confidence']:.2f}"
                cv2.putText(frame, confidence_text, (text_x_start, text_y_start + line_height), font, font_scale, text_color, 2)
                
                # Display Eye Feedback
                eye_fb_text = f"Eye: {feedback_for_frame['feedback']['eyeContact']}"
                cv2.putText(frame, eye_fb_text, (text_x_start, text_y_start + 2 * line_height), font, font_scale, text_color, 2)
                
                # Display Head Pose Feedback
                head_fb_text = f"Head: {feedback_for_frame['feedback']['headBodyAlignment']}"
                cv2.putText(frame, head_fb_text, (text_x_start, text_y_start + 3 * line_height), font, font_scale, text_color, 2)

                # Display Shoulder Alignment Feedback
                shoulder_fb_text = f"Shoulders: {feedback_for_frame['feedback']['shoulderAlignment']}"
                cv2.putText(frame, shoulder_fb_text, (text_x_start, text_y_start + 4 * line_height), font, font_scale, text_color, 2)

                # Display Head Alignment (Body) Feedback
                head_body_fb_text = f"Head & Body: {feedback_for_frame['feedback']['headBodyAlignment']}"
                cv2.putText(frame, head_body_fb_text, (text_x_start, text_y_start + 5 * line_height), font, font_scale, text_color, 2)
                
                # Display Gesture Usage Feedback (Adjust Y position due to removed spine line)
                gesture_fb_text = f"Hands: {feedback_for_frame['feedback']['hands']}"
                cv2.putText(frame, gesture_fb_text, (text_x_start, text_y_start + 6 * line_height), font, font_scale, text_color, 2)

            # Calibration prompt
            if not self._is_calibrated:
                cv2.putText(frame, "Auto-Calibrating: Please look at camera...", 
                            (frame.shape[1] // 2 - 250, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('Body Language Corrector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if display:
            self.print_final_report()
        if save_location:
            if not save_location.endswith('json'): save_location += ".json"

            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_location)
            save_data = {
                "total frames": self.total_frames,
                **self.feedback_counts, 
                'confidence proxied by eye contact': self.eye_contact_confidence['sum'] / self.total_frames
            }
            if not os.path.exists(file_path):
                with open(file_path, "x") as file:
                    json.dump(save_data, fp=file)
            else:
                with open(file_path, "w") as file:
                    json.dump(save_data, fp=file)
            

    def print_final_report(self):
        """Generate a human-readable report of the video analysis."""
        print("\n" + "="*50)
        print(f"Body Language Analysis Report ({self.total_frames} frames analyzed)")
        print("="*50)

        if self.total_frames == 0:
            print("No frames were analyzed during this session.")
            return

        # Calculate average eye contact confidence
        avg_confidence = self.eye_contact_confidence["sum"] / self.eye_contact_confidence["count"]
        print("\n--- Eye Contact ---")
        print(f"Average confidence score: {avg_confidence:.2f}")
        
        for category in ["eyeContact", "shoulderAlignment", "headBodyAlignment", "hands"]:
            print(f"\n--- {category} ---")
            if not self.feedback_counts[category]:
                print("  No data collected")
                continue
                
            for feedback, count in self.feedback_counts[category].items():
                if feedback != "Confidence score":  # Skip confidence, we showed it above
                    percentage = (count / self.total_frames) * 100
                    print(f"  {feedback}: {percentage:.1f}% ({count} frames)")
        
        print("\n" + "="*50)
        print("Analysis Complete.")
        print("="*50)


# Usage example
if __name__ == "__main__":
    corrector = BodyLanguageCorrector()
    corrector.run_video_analysis(0, show_viz=True, display=False, save_location="hello.json")