# Calibration

Calibration aligns the eye-contact logic to **your device + your face**.  
In TruthSense’s video pipeline this happens **automatically on the first frame where a face is detected** and establishes a personalized “forward gaze” baseline.

---

## What gets calibrated?

- **Reference gaze vector** → `_reference_mean`  
  A normalized 3-D vector computed from both eyes (via MediaPipe Face Landmarker + iris centers).  
- **Calibration flag** → `_is_calibrated`  
  Tracks whether the baseline has been set for the current session.

Both are stored on the `BodyLanguageCorrector` instance.

---

## When does calibration happen?

On the **first frame** with a valid face result inside `eye_and_head_analysis(...)`:

- The current gaze vector is computed from iris and eye-corner landmarks.
- `_reference_mean` is set to that vector.
- `_is_calibrated` is flipped to `True`.
- The frame is treated as **perfect eye contact** for initialization.
- A **green gaze arrow** is prepared in `viz_data` (for overlays).

### Code (excerpt)

```python
# posture.py → BodyLanguageCorrector.eye_and_head_analysis(...)
if face_landmarker_result.face_landmarks:
    landmarks = face_landmarker_result.face_landmarks[0]
    current_gaze_vector = self._get_gaze_vector_from_facelandmarks(landmarks, w, h)

    if not self._is_calibrated:
        self._reference_mean = current_gaze_vector
        self._is_calibrated = True
        # set initial green arrow viz
        viz_data["iris_center_combined"] = ...
        viz_data["gaze_endpoint"] = ...
        viz_data["gaze_color"] = (0, 255, 0)  # green
        return {
            "looking_at_camera": True,
            "iris_in_bounds": True,
            "gaze_vector_aligned": True,
            "confidence": 1.0,
            "eye_feedback_subtype": "Eye contact maintained",
            "head_feedback_subtype": "Head centered",
            "viz_data": viz_data
        }
```
## How is the gaze vector computed?

The helper function `_get_gaze_vector_from_facelandmarks(...)` uses **iris centers** and **eye corners** from MediaPipe’s Face Landmarker:

- **Left eye corner index**: `33`  
- **Right eye corner index**: `263`  
- **Left iris center**: `468`  
- **Right iris center**: `473`  

The landmark indices have been extracted from [MediaPipe Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).  
Please refer to the official docs to understand how and why the indices were chosen.
It builds eye-wise vectors (**corner → iris**), averages them, and normalizes:

```python
def _get_gaze_vector_from_facelandmarks(self, landmarks, w, h):
    def _vector_between(corner_idx, iris_obj):
        corner = np.array([
            landmarks[corner_idx].x * w,
            landmarks[corner_idx].y * h,
            landmarks[corner_idx].z * w
        ])
        iris = np.array([
            iris_obj.x * w,
            iris_obj.y * h,
            iris_obj.z * w
        ])
        vec = iris - corner
        return vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else np.zeros(3)

    left_vec  = _vector_between(33, landmarks[self.LEFT_IRIS_CENTER_IDX])   # 468
    right_vec = _vector_between(263, landmarks[self.RIGHT_IRIS_CENTER_IDX]) # 473
    avg_vec = (left_vec + right_vec) / 2
    return avg_vec / np.linalg.norm(avg_vec) if np.linalg.norm(avg_vec) != 0 else np.zeros(3)
```

## Thresholds & Device Tips

These constants live in `__init__`, so you can tune them per setup:

```python
# Blink detection (Eye Aspect Ratio)
self.EAR_THRESHOLD = 0.2

# Iris-circle radius (derived from eyelid top/bottom distance)
self.RADIUS_MULTIPLIER = 0.2   # good default for 16" laptops
# Tip: use 0.15 on 14" laptops

# Gaze alignment threshold (dot product with reference)
self.THRESHOLD_COSINE = 0.9
```
### Explanation
- **EAR_THRESHOLD** → controls blink sensitivity.  
- **RADIUS_MULTIPLIER** → affects “iris inside circle” checks (eye-in-bounds).  
- **THRESHOLD_COSINE** → controls how strict “looking at camera” detection is.  

### What the overlays mean (when `show_viz=True`)

- **Eye circles & iris dots**  
  **Green** = iris in bounds (good)  
  **Red** = iris out of bounds  

- **Gaze arrow**  
  **Green** = aligned with `_reference_mean`  
  **Red** = off-axis  

#### Code Reference (`posture.py` → drawing in `run_video_analysis`)
```python
# Eye circles colored by status
cv2.circle(..., (0, 255, 0) or (0, 0, 255), 2)  

# Gaze arrow
cv2.arrowedLine(
    frame, 
    viz_data["iris_center_combined"], 
    viz_data["gaze_endpoint"], 
    viz_data["gaze_color"], 
    2
)
```

### Reset / Re-calibrate

A new `run_video_analysis(...)` call resets calibration automatically:

```python
# posture.py → run_video_analysis(...)
self._is_calibrated = False
self._reference_mean = np.array([0.0, 0.0, -1.0])
```
---
## Confidence Computation (after calibration)

Once calibration is complete, confidence is computed using two key checks:

- **Iris in bounds?**  
  Determined by the circle test per eye.

- **Gaze aligned?**  
  Checked via the dot product between the current gaze vector and `_reference_mean`.

### Confidence Logic
Confidence is calculated as a combination of these two signals:
- **Both true:** High confidence (`1.0`)  
- **One true:** Medium confidence (`0.5`)  
- **Both false:** Low confidence (`0.0`)  

Additionally, the dot product margin contributes to a **continuous confidence score**, smoothed across frames.

```python
dot_product = np.dot(current_gaze_vector, self._reference_mean)
gaze_vector_aligned = np.clip(dot_product, -1.0, 1.0) > self.THRESHOLD_COSINE

if looking_at_camera:
    confidence = 1.0
elif iris_in_bounds or gaze_vector_aligned:
    confidence = 0.5
else:
    confidence = 0.0

# refine with alignment component if iris is in bounds
confidence_gaze_component = (dot_product - self.THRESHOLD_COSINE) / (1.0 - self.THRESHOLD_COSINE)
confidence = (confidence + np.clip(confidence_gaze_component, 0, 1)) / 2 if iris_in_bounds else 0.0
```
A session-level average is maintained in self.eye_contact_confidence.

## Recommended Calibration Posture
- Sit at your normal working distance from the camera.  
- Look directly into the camera for ~1 second at stream start.  
- Prefer even lighting; avoid backlight that obscures eyes.  
- Ensure the full face and both eyes are visible (no heavy occlusion).  

---

## Troubleshooting

- **Arrow is often red even when looking at the camera**  
  → Lower `THRESHOLD_COSINE` slightly (e.g., `0.85`).  

- **Iris frequently “out of bounds”**  
  → Reduce `RADIUS_MULTIPLIER` (e.g., `0.15` on smaller screens).  
  → Improve lighting; make sure irises are clearly visible.  

- **Blinking is over-triggered**  
  → Decrease `EAR_THRESHOLD` (e.g., `0.18`).  

- **Calibration locks onto a wrong direction**  
  → Re-start analysis or set `corrector._is_calibrated = False` to re-calibrate.  

---
## Minimal Usage
```python
from video.posture import BodyLanguageCorrector

corrector = BodyLanguageCorrector()

# (Optional) tweak thresholds per device
corrector.RADIUS_MULTIPLIER = 0.15      # 14" laptop tip
corrector.THRESHOLD_COSINE = 0.88       # slightly more permissive

# Start session (first valid face frame will auto-calibrate)
corrector.run_video_analysis(
    video_source=0,     # webcam (or a path to a video file)
    show_viz=True,      # draw circles and gaze arrow
    display=True,       # print final report at the end
    save_location="session_report.json"   # optional JSON next to posture.py
)
```

---
