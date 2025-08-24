# Class Structure

This document explains the main classes used in the `video/` pipeline for body language and eye contact analysis.

---

## `BodyLanguageCorrector`

### Purpose
- Main entry point for running video analysis.
- Manages OpenCV capture loop and delegates per-frame processing.
- Aggregates session statistics and prints/saves final reports.

### Key Methods
- **`run_video_analysis(video_source=0, show_viz=False, display=True, save_location=None)`**  
  Starts webcam (or video file) feed, processes frames in real time, optionally displays visualization overlays, and saves report.

- **`process_frame(frame, timestamp)`**  
  Core per-frame analysis function. Runs face, pose, and hand landmarkers, merges results into structured outputs.

- **`print_final_report()`**  
  Generates session-level metrics (eye contact %, blink count, posture stability, gesture frequency, etc.).

---

## `EyeContactAnalyzer`

### Purpose
- Detects gaze alignment and blinking using face/iris landmarks.
- Handles **auto-calibration** from the first valid frame.

### Key Attributes
- `_reference_mean`: calibrated baseline gaze vector.
- `THRESHOLD_COSINE`: dot product threshold for gaze alignment (default 0.9).
- `EAR_THRESHOLD`: blink detection threshold (default 0.2).
- `RADIUS_MULTIPLIER`: used for iris circle size.

### Key Methods
- **`analyze(face_landmarks)`** → returns per-frame eye-contact confidence, blink events, and gaze vector status.

---

## `PostureAnalyzer`

### Purpose
- Tracks upper-body posture using pose landmarks.
- Focuses on **head tilt** and **shoulder alignment**.

### Key Methods
- **`analyze(pose_landmarks)`** → returns posture classification (centered, tilted, slouched).
- **`compute_angle(head, shoulders)`** → helper for tilt estimation.

---

## `GestureAnalyzer`

### Purpose
- Detects hand positions and movements.
- Categorizes gestures into **low / mid / high** relative to body.

### Key Methods
- **`analyze(hand_landmarks)`** → returns presence and class of gesture for each frame.
- **`track_frequency()`** → counts gesture occurrences over session.

---

## `ReportAggregator`

### Purpose
- Maintains counters and accumulates results from all analyzers.
- Consolidates session into final JSON output.

### Key Methods
- **`update(frame_results)`** → merge per-frame results into counters.
- **`to_json(filepath)`** → save session report in JSON format.

---

# Summary
The pipeline follows a **modular class structure**:
- `BodyLanguageCorrector` orchestrates flow.
- `EyeContactAnalyzer`, `PostureAnalyzer`, and `GestureAnalyzer` handle domain-specific tasks.
- `ReportAggregator` merges outputs into reports.

This design ensures flexibility: each analyzer can be tuned or replaced independently without breaking the full pipeline.