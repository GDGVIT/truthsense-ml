# Posture Analysis Module Documentation

This document explains the **posture analysis** functionality implemented in the `BodyLanguageCorrector` class. It covers shoulder alignment, head-to-body orientation, thresholds used, feedback categorization, and reporting.

---

## Overview

The module leverages **MediaPipe Pose Landmarker** to analyze:

- **Shoulder alignment** (horizontal balance)  
- **Head-to-body alignment** (nose vs. shoulder center)  
- **Head tilt detection**  
- **Pose stability across frames**  

These outputs provide real-time feedback for presentation coaching, posture correction, and confidence tracking.

---

## Key Components

### Shoulder Alignment

- Uses **left (11)** and **right (12)** shoulder landmarks.  
- Computes vertical difference:  
  `Δy = |y_left − y_right|`.  
- Categorization thresholds:  
  - `< 0.05` → *Good shoulder alignment*  
  - `< 0.10` → *Slight shoulder tilt*  
  - `≥ 0.10` → *Shoulders are tilted*  

This metric highlights slouching or uneven posture.

---

### Head-to-Body Alignment

- Uses **nose (0)** landmark relative to shoulder center:  
  `shoulder_center = ( (x_left + x_right)/2 , (y_left + y_right)/2 )`.  
- Horizontal deviation:  
  `Δx = |nose_x − shoulder_center_x|`.  
- Categorization thresholds:  
  - `< 0.03` → *Head aligned with body*  
  - `< 0.06` → *Head slightly tilted*  
  - `≥ 0.06` → *Head tilted*  

This measures whether the speaker’s head is centered with their torso.

---

### Combined Feedback

- Both shoulder and head metrics update **feedback counters** per frame.  
- Results accumulate into the session report (`feedback_counts`) as subtype frequencies.  

---

## Feedback Subtypes

- **Shoulder Alignment**
  - *Good shoulder alignment*
  - *Slight shoulder tilt*
  - *Shoulders are tilted*
  - *No pose detected*

- **Head-Body Alignment**
  - *Head aligned with body*
  - *Head slightly tilted*
  - *Head tilted*
  - *No pose detected*

---

## Visualization Data

When `show_viz=True`, posture analysis provides:

- Landmark positions (shoulders + nose)  
- Horizontal reference line across shoulders  
- HUD text overlay:  
  - `"Shoulders aligned"` vs `"Tilted"`  
  - `"Head centered"` vs `"Head tilted"`  

---

## Example Output

```json
{
  "shoulderAlignment": "Slight shoulder tilt",
  "headBodyAlignment": "Head slightly tilted",
  "landmarks": {
    "left_shoulder": [250, 380],
    "right_shoulder": [420, 382],
    "nose": [335, 180]
  },
  "metrics": {
    "shoulder_delta_y": 0.072,
    "head_delta_x": 0.041
  }
}
```

---
## Parameters

| Parameter    | Thresholds               | Description                                 |
|--------------|--------------------------|---------------------------------------------|
| Shoulder Δy  | <0.05, <0.10, otherwise  | Vertical difference between shoulders       |
| Head Δx      | <0.03, <0.06, otherwise  | Horizontal offset of nose vs. shoulder center |

---

## Usage

```python
corrector = BodyLanguageCorrector()
pose_result = corrector.posture_analysis(pose_landmarks)

print(pose_result["shoulderAlignment"], pose_result["headBodyAlignment"])
```

## Error Handling & Edge Cases

- If **no pose is detected**, both categories return `"No pose detected"`.  
- Small jitter across frames is handled by frame-based counters.  
- Requires both shoulders visible; partial framing reduces accuracy.  

---

## Reports & Saving

In the session summary, posture analysis contributes entries such as:
```json
"shoulderAlignment": {
  "Good shoulder alignment": 1100,
  "Slight shoulder tilt": 100,
  "Shoulders are tilted": 34
},
"headBodyAlignment": {
  "Head aligned with body": 900,
  "Head slightly tilted": 270,
  "Head tilted": 64
}
```
Percentages are derived automatically from total frames.
---
## Best Practices

- Keep **upper body fully visible** (head + shoulders).
- Sit or stand in a **neutral upright position**.
- Avoid leaning too close to the camera.

---

## Limitations

- Assumes **frontal orientation**; side profiles may misclassify.
- Sensitive to **partial visibility** (one shoulder cropped).
- Uses **2D keypoints only**, no depth correction.

---

## References

- [MediaPipe Pose Landmarker Docs](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

---

## Next Steps

- Add **smoothing filters** (e.g., moving averages or Kalman filters).
- Explore **3D pose estimation** for more robust tilt detection.
- Provide **real-time corrective prompts** (e.g., “Straighten your shoulders”).
