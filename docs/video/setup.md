# Setup Guide

This guide explains how to install dependencies, prepare model files, and run the **BodyLanguageCorrector** video analysis pipeline.

---

## 1. Environment Setup

### Python Version
- Python **3.9-3.11** is recommended (tested on 3.10).

### Required Libraries
Install dependencies using `pip`:

```bash
pip install opencv-python mediapipe numpy
```
If you are saving results to JSON:
```bash
pip install jsonschema
```
---

## 2. Model Files

The code expects MediaPipe task files to be present in:
```
/posenet_models/
├── pose_landmarker.task
├── face_landmarker.task
└── hand_landmarker.task
```
- Download them from the official [MediaPipe Models repository](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).
- Place them inside `posenet_models` at the project root or adjust the paths in `posture.py`.
---
## 3. Directory Structure

**Typical layout:**
```
project_root/
├── posture/
│   └── posture.py
│   └── …
│
├── posenet_models/
│   ├── pose_landmarker.task
│   ├── face_landmarker.task
│   └── hand_landmarker.task
│
└── outputs/
    └── hello.json    # Example saved report
```
---
## 4. Running the Analysis

### Webcam Input
```bash
python posture/posture.py
```

### Video File Input
Edit the call at the bottom of posture.py:
```python
if __name__ == "__main__":
    corrector = BodyLanguageCorrector()
    corrector.run_video_analysis(0, show_viz=True, save_location="outputs/session.json")
```
---
## 5. Key Options

- **video_source**  
    - `0` → default webcam  
    - `"path/to/video.mp4"` → video file  
- **show_viz=True** → Draw overlays (eye circles, gaze arrows, text feedback).  
- **display=True** → Print final report in terminal.  
- **save_location="outputs/file.json"** → Save session summary.  

---
## 6. Output

- At the end of a run, a JSON file (if `save_location` is set) contains:
  - Total frames analyzed  
  - Category counts (`eyeContact`, `shoulderAlignment`, etc.)  
  - Average eye contact confidence  

**Example:**
```json
{
  "total frames": 1243,
  "eyeContact": { "Eye contact maintained": 823, "Blinking": 50 },
  "shoulderAlignment": { "Good shoulder alignment": 950 },
  "headBodyAlignment": { "Head aligned with body": 912 },
  "hands": { "Both hands in frame": 411 },
  "confidence proxied by eye contact": 0.87
}
```
---
## 7. Tips

- Calibrate by looking at the camera for ~1s when the stream starts.  
- Use **even lighting** for better iris/eye detection.  
- To re-calibrate mid-run:  
```python
corrector._is_calibrated = False
```
(next detected face frame resets the reference gaze).
---