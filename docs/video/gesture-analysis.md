# Gesture Analysis Module

The **Gesture Analysis** module evaluates the presence and relative position of hands during a video session. It uses Mediapipe’s **HandLandmarker** outputs to categorize gestures and provide structured feedback.

---

## Features

- Detects if hands are **absent**, **single**, or **both hands present** in frame.  
- Classifies hand position as **high**, **middle**, or **low** relative to frame height.  
- Provides structured feedback strings and updates session-level feedback counters.  
- Integrates seamlessly with the overall `BodyLanguageCorrector` class.

---

## Parameters

| Parameter         | Default | Description |
|-------------------|---------|-------------|
| Frame Height Cutoffs | 0.5, 0.9 | Defines thresholds for hand height classification (High, Middle, Low). |

---

## Outputs

- `"Hands not in frame"`
- `"One hand in frame; Hand high in the frame"`
- `"One hand in frame; Hand in the middle of the frame"`
- `"One hand in frame; Hand low in the frame"`
- `"Both hands in frame; One hand high in the frame; One hand low in the frame"` (etc., depending on detection)

---

## Example Usage

```python
# Assume `hand_landmarks_list` comes from Mediapipe's HandLandmarker
gestures_feedback = self.gesture_analysis(hand_landmarks_list)
print(gestures_feedback)
# Example output: "Both hands in frame; One hand high in the frame; One hand low in the frame"
```

---
## Notes
- The analysis is **relative to the frame height**; ensure consistent camera framing.  
- Gesture feedback is primarily **qualitative** and complements posture/eye-contact analysis.  
- Can be extended with more detailed hand-shape recognition (e.g., **open palm**, **fist**, **pointing**).  
---