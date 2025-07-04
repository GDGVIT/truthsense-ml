import os
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional

class BodyLanguageCorrector:
    def __init__(self):
        # Initialize MediaPipe tasks
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseDetector = mp.tasks.vision.PoseLandmarker
        self.PoseDetectorOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.FaceDetector = mp.tasks.vision.FaceLandmarker
        self.FaceDetectorOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        pose_path = os.path.join(BASE_DIR, '../mediapipe_models/pose_landmarker.task')
        face_path = os.path.join(BASE_DIR, '../mediapipe_models/face_landmarker.task')
        hand_path = os.path.join(BASE_DIR, '../mediapipe_models/hand_landmarker.task')
        
        # Initialize pose detector
        pose_options = self.PoseDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=pose_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_detector = self.PoseDetector.create_from_options(pose_options)
        

        # Initialize face detector
        face_options = self.FaceDetectorOptions(
            base_options=self.BaseOptions(model_asset_path=face_path),
            running_mode=self.VisionRunningMode.VIDEO,
            # min_suppression_threshold=0.3
        )
        self.face_detector = self.FaceDetector.create_from_options(face_options)

        
       
        # Initialize hand landmarker
        hand_options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=hand_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = self.HandLandmarker.create_from_options(hand_options)
        
        # Analysis parameters
        self.eye_contact_history = []
        self.eye_contact_window = 10
        self.posture_history = []
        self.gesture_history = []
        self.feedback_window = 30
        
        # Timing
        self.last_analysis_time = 0
        self.analysis_interval = 0.5  # 2 times per second
        self.frame_count = 0

    def detect_eye_contact(self, image: np.ndarray, timestamp_ms: int) -> Dict:
        """
        Detect if user is maintaining eye contact with camera using face landmarks and vector projection.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        face_landmarker_result = self.face_detector.detect_for_video(mp_image, timestamp_ms)

        if not face_landmarker_result.face_landmarks:
            return {
                "looking_at_camera": False,
                "confidence": 0.0,
                "feedback": "No face detected"
            }
        
        landmarks = face_landmarker_result.face_landmarks[0]
        
        # --- NEW: Vector Projection Logic ---

        def get_gaze_ratio(eye_points, iris_point):
            """
            Calculates the gaze ratio using vector projection.
            A ratio of ~0.5 means looking straight.
            """
            # Define vectors
            eye_corner_medial = eye_points[0]
            eye_corner_lateral = eye_points[1]
            
            # Vector for the full eye width
            v_eye = (eye_corner_lateral.x - eye_corner_medial.x, eye_corner_lateral.y - eye_corner_medial.y)
            # Vector from medial corner to the iris
            v_iris = (iris_point.x - eye_corner_medial.x, iris_point.y - eye_corner_medial.y)
            # Dot product to project iris vector onto the eye vector
            dot_product = (v_eye[0] * v_iris[0]) + (v_eye[1] * v_iris[1])
            # Squared magnitude of the eye vector
            mag_sq_eye = (v_iris[0]**2) + (v_iris[1]**2)
            
            if mag_sq_eye == 0:
                return 0.5 # Avoid division by zero, return center

            # The ratio is the projected length over the total length
            ratio = dot_product / mag_sq_eye
            return ratio

        # Standard landmark indices
        LEFT_EYE_POINTS = [landmarks[33], landmarks[133]]
        RIGHT_EYE_POINTS = [landmarks[263], landmarks[362]]
        LEFT_IRIS_POINT = landmarks[473]
        RIGHT_IRIS_POINT = landmarks[468]
                
        left_ratio = get_gaze_ratio(LEFT_EYE_POINTS, LEFT_IRIS_POINT)
        right_ratio = get_gaze_ratio(RIGHT_EYE_POINTS, RIGHT_IRIS_POINT)

        print(left_ratio, right_ratio)

        # Heuristic: If the iris is in the middle ~40% of the eye's projected axis.
        EYE_CONTACT_THRESHOLD_MIN = 0.30
        EYE_CONTACT_THRESHOLD_MAX = 0.70
        
        looking_at_camera = (EYE_CONTACT_THRESHOLD_MIN < left_ratio < EYE_CONTACT_THRESHOLD_MAX and
                             EYE_CONTACT_THRESHOLD_MIN < right_ratio < EYE_CONTACT_THRESHOLD_MAX)

        # Confidence score based on how close the ratios are to the center (0.5)
        avg_ratio_distance_from_center = (abs(left_ratio - 0.5) + abs(right_ratio - 0.5))
        eye_contact_score = max(0, 1 - avg_ratio_distance_from_center * 2) # Multiplier adjusted for better scoring

        # Update history and generate feedback
        self.eye_contact_history.append(looking_at_camera)
        if len(self.eye_contact_history) > self.eye_contact_window:
            self.eye_contact_history.pop(0)
        
        avg_eye_contact = sum(self.eye_contact_history) / len(self.eye_contact_history) if self.eye_contact_history else 0
        
        if not looking_at_camera:
             feedback = "Poor eye contact - look at the camera"
        elif avg_eye_contact > 0.8:
            feedback = "Excellent eye contact!"
        elif avg_eye_contact > 0.6:
            feedback = "Good eye contact"
        else:
            feedback = "Good, keep looking at the camera"
        
        return {
            "looking_at_camera": looking_at_camera,
            "confidence": eye_contact_score,
            "average_eye_contact": avg_eye_contact,
            "feedback": feedback
        }
           
    def analyze_body_language(self, image: np.ndarray, timestamp_ms: int) -> Dict:
        """
        Analyze body language for public speaking feedback
        """
        # try:
            # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Detect pose
        pose_result = self.pose_detector.detect_for_video(mp_image, timestamp_ms)
        
        # Detect hands
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        feedback = []
        scores = {}
        
        # Analyze posture if pose detected
        if pose_result.pose_landmarks:
            posture_analysis = self._analyze_posture(pose_result.pose_landmarks[0])
            feedback.extend(posture_analysis["feedback"])
            scores.update(posture_analysis["scores"])
        else:
            feedback.append("No pose detected")
        
        # Analyze hand gestures if hands detected
        if hand_result.hand_landmarks:
            gesture_analysis = self._analyze_hand_gestures(hand_result.hand_landmarks)
            feedback.extend(gesture_analysis["feedback"])
            scores.update(gesture_analysis["scores"])
        
        # Calculate overall body language score
        overall_score = np.mean(list(scores.values())) if scores else 0.0
        
        return {
            "overall_score": overall_score,
            "scores": scores,
            "feedback": feedback,
            "pose_detected": bool(pose_result.pose_landmarks),
            "hands_detected": len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        }
            
        # except Exception as e:
        #     return {
        #         "overall_score": 0.0,
        #         "scores": {},
        #         "feedback": [f"Error in body language analysis: {str(e)}"],
        #         "pose_detected": False,
        #         "hands_detected": 0
        #     }

    def _analyze_posture(self, pose_landmarks) -> Dict:
        """
        Analyze posture for public speaking
        """
        feedback = []
        scores = {}
        
        # Get key landmarks
        # landmarks = pose_landmarks.landmark
        landmarks = pose_landmarks
        
        # Shoulder alignment
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        
        if shoulder_slope < 0.05:
            feedback.append("Good shoulder alignment")
            scores["shoulder_alignment"] = 0.9
        elif shoulder_slope < 0.1:
            feedback.append("Slight shoulder tilt - try to keep shoulders level")
            scores["shoulder_alignment"] = 0.7
        else:
            feedback.append("Shoulders are tilted - straighten your posture")
            scores["shoulder_alignment"] = 0.4
        
        # Head position
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        
        head_alignment = abs(nose.x - shoulder_center_x)
        
        if head_alignment < 0.03:
            feedback.append("Good head alignment")
            scores["head_alignment"] = 0.9
        elif head_alignment < 0.06:
            feedback.append("Head slightly off-center")
            scores["head_alignment"] = 0.7
        else:
            feedback.append("Keep your head centered over your shoulders")
            scores["head_alignment"] = 0.4
        
        # Back straightness (using hip to shoulder distance)
        left_hip = landmarks[23]
        spine_straightness = abs(left_shoulder.z - left_hip.z)
        
        if spine_straightness < 0.1:
            feedback.append("Good posture - standing straight")
            scores["spine_straightness"] = 0.9
        elif spine_straightness < 0.2:
            feedback.append("Straighten your back a bit more")
            scores["spine_straightness"] = 0.6
        else:
            feedback.append("Stand up straighter - avoid slouching")
            scores["spine_straightness"] = 0.3
        
        return {"feedback": feedback, "scores": scores}

    def _analyze_hand_gestures(self, hand_landmarks_list) -> Dict:
        """
        Analyze hand gestures for public speaking
        """
        feedback = []
        scores = {}
        
        num_hands = len(hand_landmarks_list)
        
        if num_hands == 0:
            feedback.append("Use hand gestures to enhance your speech")
            scores["gesture_usage"] = 0.5
        elif num_hands == 1:
            feedback.append("Good use of gestures - consider using both hands")
            scores["gesture_usage"] = 0.7
        else:
            feedback.append("Great use of both hands for gesturing")
            scores["gesture_usage"] = 0.9
        
        # Analyze gesture positioning
        for i, hand_landmarks in enumerate(hand_landmarks_list):
            hand_side = "right" if i == 0 else "left"
            
            # Check if hands are in visible gesture zone (not too low, not too high)
            wrist = hand_landmarks[0]
            
            # Gesture height analysis
            if wrist.y < 0.3:  # Too high
                feedback.append(f"Lower your {hand_side} hand slightly")
                scores[f"{hand_side}_hand_position"] = 0.6
            elif wrist.y > 0.8:  # Too low
                feedback.append(f"Raise your {hand_side} hand for better visibility")
                scores[f"{hand_side}_hand_position"] = 0.6
            else:
                scores[f"{hand_side}_hand_position"] = 0.9
        
        return {"feedback": feedback, "scores": scores}

    def process_frame(self, frame: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Process a single frame and return analysis results
        """
        current_time = time.time()
        self.frame_count += 1
        timestamp_ms = int(current_time * 1000)
        
        # Only analyze every 0.5 seconds (2 times per second)
        # if current_time - self.last_analysis_time >= self.analysis_interval:
        eye_contact_result = self.detect_eye_contact(frame, timestamp_ms)
        body_language_result = self.analyze_body_language(frame, timestamp_ms)
        self.last_analysis_time = current_time
        
        return eye_contact_result, body_language_result
        
        # Return empty results for frames that aren't analyzed
        # return {}, {}

    def run_video_analysis(self, video_source=0):
        """
        Run real-time video analysis
        video_source: 0 for webcam, or path to video file
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting real-time body language analysis...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            eye_contact_result, body_language_result = self.process_frame(frame)
            
            # Display results on frame
            if eye_contact_result:
                # Display eye contact feedback
                cv2.putText(frame, f"Eye Contact: {eye_contact_result.get('feedback', '')}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                confidence = eye_contact_result.get('confidence', 0)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if body_language_result:
                # Display body language feedback
                feedback_list = body_language_result.get('feedback', [])
                y_offset = 90
                for feedback in feedback_list[:3]:  # Show max 3 feedback items
                    cv2.putText(frame, feedback, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    y_offset += 25
                
                overall_score = body_language_result.get('overall_score', 0)
                cv2.putText(frame, f"Overall Score: {overall_score:.2f}", 
                           (10, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Show frame
            cv2.imshow('Body Language Corrector', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    # Initialize the corrector
    corrector = BodyLanguageCorrector()
    
    # Run real-time analysis
    corrector.run_video_analysis(0)  # Use webcam
    
    # Alternative: Process single frame
    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # if ret:
    #     eye_contact, body_language = corrector.process_frame(frame)
    #     print("Eye Contact:", eye_contact)
    #     print("Body Language:", body_language)
    # cap.release()