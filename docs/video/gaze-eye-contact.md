# Gaze & Eye Contact Module Documentation

This document explains the **gaze and eye-contact analysis** functionality implemented in the `BodyLanguageCorrector` class. It covers calibration, gaze vector computation, iris detection, blink detection, posture, gestures, and confidence scoring.

---

## Overview

The module leverages **MediaPipe Landmarkers** to analyze:

* **Eye contact maintenance**
* **Head alignment**
* **Shoulder alignment**
* **Hand gestures**
* **Blink detection**
* **Gaze vector direction**

These outputs are used to provide real-time feedback for presentation coaching, posture correction, and confidence tracking.

---

## Key Components

### Calibration

* On the **first detected frame**, the module auto-calibrates by saving the initial gaze vector as `_reference_mean`.
* Until calibration is complete:

  * The system assumes perfect eye contact.
  * Confidence is set to **1.0**.
  * A yellow HUD prompt can be shown: *“Auto‑Calibrating: Please look at camera…”*

### Gaze Vector

* Computed as the **average of two vectors**:

  * Left eye: medial corner → iris center
  * Right eye: medial corner → iris center
* Normalized to a unit vector for direction consistency.
* Compared against the calibrated `_reference_mean` using a **cosine similarity threshold**.

### Iris In-Bounds Check

* Each eye is modeled as a **circle region** derived from eyelid landmarks.
* The **iris center pixel** is checked if it lies inside this circle.
* If both eyes are in-bounds → considered valid eye tracking.
* Used to detect whether the user is maintaining eye contact.

### Blink Detection

* Uses **Eye Aspect Ratio (EAR)**:

  * EAR = (vertical distance) / (horizontal distance)
* If EAR < `EAR_THRESHOLD` (default 0.2) for both eyes:

  * Marks frame as **Blinking**.
  * Eye contact not counted.
  * Eye vector drawing is skipped.

### Confidence Score

* Confidence combines:

  1. **Iris in-bounds check**
  2. **Gaze vector alignment**
* Values:

  * `1.0` → Strong eye contact
  * `0.5` → Partial (either aligned gaze or in-bounds)
  * `0.0` → No contact
* Running average tracked across frames using `eye_contact_confidence`.

---

## Posture & Gesture Analysis

### Shoulder Alignment

* Uses Pose Landmarker indices `11 (left)` and `12 (right)`.
* Horizontal alignment test:

  * `|y_L – y_R| < 0.05` → Good
  * `< 0.1` → Slight tilt
  * Otherwise → Tilted

### Head Alignment

* Uses `nose (0)` vs shoulder center.
* Horizontal offset:

  * `< 0.03` → Head aligned
  * `< 0.06` → Slight tilt
  * Otherwise → Tilted

### Gesture Analysis

* Counts number of hands detected by Hand Landmarker.
* Labels:

  * **No hands**
  * **One hand** (high / middle / low based on wrist.y)
  * **Both hands** (each categorized)

---

## Feedback Subtypes

* **Eye Contact**

  * *Eye contact maintained*
  * *Eyes off-center*
  * *Blinking*
  * *No face detected*

* **Head Alignment**

  * *Head aligned with body*
  * *Head slightly tilted*
  * *Head tilted*
  * *No pose detected*

* **Shoulder Alignment**

  * *Good shoulder alignment*
  * *Slight shoulder tilt*
  * *Shoulders are tilted*
  * *No pose detected*

* **Hands**

  * *Hands not in frame*
  * *One hand in frame; high/middle/low*
  * *Both hands in frame; …*

---

## Visualization Data

Returned with every frame when `show_viz=True`:

* Eye circles (center + radius)
* Iris pixel positions
* Combined iris midpoint
* Gaze vector endpoint
* Color:

  * Green → aligned gaze
  * Red → misaligned gaze

This `viz_data` structure is designed for frontend overlays.

---

## Example Output

```json
{
  "looking_at_camera": true,
  "iris_in_bounds": true,
  "gaze_vector_aligned": true,
  "confidence": 0.92,
  "eye_feedback_subtype": "Eye contact maintained",
  "head_feedback_subtype": "Head aligned with body",
  "viz_data": {
    "left_eye_center": [310, 220],
    "left_radius": 15,
    "right_eye_center": [410, 220],
    "right_radius": 15,
    "left_iris_pixel": [315, 222],
    "right_iris_pixel": [408, 223],
    "iris_center_combined": [362, 221],
    "gaze_endpoint": [362, 301],
    "gaze_color": [0, 255, 0]
  }
}
```

---

## Parameters

| Parameter           | Default | Description                                          |
| ------------------- | ------- | ---------------------------------------------------- |
| `EAR_THRESHOLD`     | 0.2     | Blink detection sensitivity                          |
| `RADIUS_MULTIPLIER` | 0.2     | Circle size scaling factor (varies with screen size) |
| `THRESHOLD_COSINE`  | 0.9     | Cosine similarity cutoff for gaze alignment          |

---

## Reporting & Saving

* All subtypes are accumulated in `feedback_counts`.
* `print_final_report()` prints:

  * Total frames
  * Average eye-contact confidence
  * % of frames per subtype
* If `save_location` is set, a JSON report is written with these results.

---

## Usage

```python
corrector = BodyLanguageCorrector()
result = corrector.eye_and_head_analysis(frame, timestamp)
print(result["eye_feedback_subtype"], result["confidence"])
```

---

## Error Handling & Edge Cases

* **No face/pose detected** → Subtype logged as “No face/pose detected”.
* **Blink detected** → Frame marked as Blinking, vectors skipped.
* **Zero-norm gaze vector** → Falls back to safe defaults.

---

## Device Tuning

| Parameter           | Purpose                                     | Typical values                               |
| ------------------- | ------------------------------------------- | -------------------------------------------- |
| `RADIUS_MULTIPLIER` | Eye circle radius as multiple of eye height | **0.20** (16″ laptop), **0.15** (14″ laptop) |
| `THRESHOLD_COSINE`  | Cosine similarity for gaze alignment        | 0.85–0.95 (default 0.90)                     |
| `EAR_THRESHOLD`     | Blink detection threshold                   | 0.18–0.25 (default 0.20)                     |

---

## Limitations

* **Single Face Only**: Current version assumes one face.
* **2D Iris Approximation**: Circle heuristic may fail under extreme head turns.
* **Blink Ambiguity**: Fast or partial blinks can be misclassified.
* **Lighting Sensitivity**: Performance degrades with poor webcam quality, glare, or low light.

---

## Extensibility / Next Steps

* Integrate **Kalman filters** or moving averages for smoother gaze stability.
* Add **multi-person support**.
* Experiment with **3D head pose estimation**.
* Improve blink classification with temporal smoothing.

---

## Reference Landmark Indices

* **Face Landmarker**:

  * Iris centers: `468 (left)`, `473 (right)`
  * Eye corners: left `33, 133`; right `263, 362`
  * EAR: left `159, 145, 33, 133`; right `386, 374, 362, 263`
* **Pose Landmarker**:

  * Shoulders: `11 (left)`, `12 (right)`
  * Nose: `0`

---

**End of Documentation**
