# Frontend Interface

This document describes the contract between the frontend and the fusion/LLM service.  
It specifies what the frontend must **send** (posture features) and what it will **receive** (structured feedback used to render UI).

---

## What the Frontend Sends

The frontend supplies a **posture_features** object summarizing the session’s non-verbal signals.  
It must follow this shape (keys are required; values are numbers):

```json
{
  "eyeContact":        { "Eye contact maintained": 0.72, "Blinking": 0.06, "Eyes off-center": 0.22 },
  "shoulderAlignment": { "Good shoulder alignment": 0.64, "Slight shoulder tilt": 0.28, "Shoulders are tilted": 0.08 },
  "headBodyAlignment": { "Head aligned with body": 0.70, "Head slightly tilted": 0.20, "Head tilted": 0.10 },
  "handGestures":      { "Both hands in frame": 0.45, "One hand high in the frame": 0.15, "Hands not in frame": 0.40 }
}
```

---

## Field definitions

- **eyeContact: { [label: string]: number }**  
  - Typical labels: `"Eye contact maintained"`, `"Blinking"`, `"Eyes off-center"`.

- **shoulderAlignment: { [label: string]: number }**  
  - Typical labels: `"Good shoulder alignment"`, `"Slight shoulder tilt"`, `"Shoulders are tilted"`.

- **headBodyAlignment: { [label: string]: number }**  
  - Typical labels: `"Head aligned with body"`, `"Head slightly tilted"`, `"Head tilted"`.

- **handGestures: { [label: string]: number }**  
  - Typical labels: `"Both hands in frame"`, `"One hand high in the frame"`,  
    `"One hand in the middle of the frame"`, `"One hand low in the frame"`, `"Hands not in frame"`.

---

### Value semantics

- Values may be **fractions (0–1)** or **counts**. Fractions are preferred.  
- If counts are provided, they should be comparable across categories (e.g., frame counts).  
- The backend does not assume the values sum to 1; send what best represents your session summary.  

> Note: The exact labels you use are surfaced verbatim in the LLM prompt (they don’t need to match a fixed enumeration). Use clear, human-readable labels.

---

## What the Frontend Receives

The service returns a JSON string matching the `FrontendResponse` schema below, plus:

- **speaking_rate** — integer words per minute (WPM)  
- **overall_score** — integer 0–100 (weighted composite)  
- **transcript** — full ASR transcript

### Response Schema(for rendering)
```json
type FrontendResponse = {
  transcript: string;          // full-text transcript
  overall_score: number;       // 0..100
  speaking_rate: number;       // words per minute (WPM)

  fluency_evaluator: {
    comment: string;           // 3–5 sentence summary of flow/pauses/fillers/pace
    fluency_score: number;             // 0..100 (fluency)
  };

  language_evaluator: {
    strengths: string[];       // 2–5 items
    improvements: string[];    // 2–5 items
    structure_score: number;   // 0..100 (organization/flow)
    grammar_score: number;     // 0..100 (correctness)
  };

  speech_evaluator: {
    strengths: string[];       // 2–5 items (clarity, delivery, confidence)
    improvements: string[];    // 2–5 items
    clarity_score: number;     // 0..100
    confidence_score: number;  // 0..100
  };

  posture_evaluator: {
    tips: string[];            // 3–7 actionable posture/gesture pointers
    posture_score: number;     // 0..100
  };
};
```

### Example Response
```json
{
  "transcript": "Good afternoon everyone, today I'll walk you through...",
  "overall_score": 82,
  "speaking_rate": 156,
  "fluency_evaluator": {
    "comment": "Pace stays steady with natural pausing. Minimal fillers; flow is cohesive with clear handoffs between ideas.",
    "fluency_score": 84
  },
  "language_evaluator": {
    "strengths": [
      "Clear structure with preview and wrap-up",
      "Concise phrasing",
      "Appropriate vocabulary for the audience"
    ],
    "improvements": [
      "Tighten transitions between sections two and three",
      "Add one concrete example to support the claim on latency"
    ],
    "structure_score": 80,
    "grammar_score": 88
  },
  "speech_evaluator": {
    "strengths": [
      "Good articulation of key terms",
      "Varied tone keeps attention"
    ],
    "improvements": [
      "Sustain projection at sentence ends",
      "Slightly widen pitch range on key points"
    ],
    "clarity_score": 83,
    "confidence_score": 79
  },
  "posture_evaluator": {
    "tips": [
      "Keep both hands in frame for emphasis on key points",
      "Square shoulders to camera before starting",
      "Hold eye contact for the first sentence of each section",
      "Relax jaw and blink naturally to avoid a fixed stare"
    ],
    "posture_score": 78
  }
}
```

---
## Minimal end-to-end request model

If you call the service from the frontend (or your server), the request must include:

- **audio_path** (or the audio bytes handled by the server)  
- **posture_features** (the object defined above)

---

### Example request body (if your server exposes a REST endpoint):

```json
{
  "audio_path": "uploads/sample.wav",
  "posture_features": {
    "eyeContact": { "Eye contact maintained": 0.85, "Blinking": 0.1, "Eyes off-center": 0.05 },
    "shoulderAlignment": { "Good shoulder alignment": 0.9, "Slight shoulder tilt": 0.1 },
    "headBodyAlignment": { "Head aligned with body": 0.75, "Head slightly tilted": 0.25 },
    "handGestures": { "Both hands in frame": 0.6, "One hand in the middle of the frame": 0.4 }
  }
}
```

---

## Validation Notes

- All evaluator scores are integers in the range **1–100**.  
- `speaking_rate` is an integer WPM.  
- Arrays `strengths`, `improvements`, and `tips` are short lists (2–7 items depending on section).  
- The backend is tolerant of posture label names; make them clear and user-facing.  

---

## Common Integration Questions

**Do posture values need to sum to 1?**  
No. Fractions are preferred, but counts are accepted.  

**Can we omit a posture category?**  
Provide all four top-level keys (`eyeContact`, `shoulderAlignment`, `headBodyAlignment`, `handGestures`).  
If a category truly doesn’t apply, pass an empty object `{}`.  

**Where do the extra fields come from?**  
- `speaking_rate` and `transcript` are computed from the audio.  
- `overall_score` is computed server-side from the evaluator sub-scores.  

---

## TypeScript helpers (optional)

```ts
export interface PostureFeatures {
  eyeContact: Record<string, number>;
  shoulderAlignment: Record<string, number>;
  headBodyAlignment: Record<string, number>;
  handGestures: Record<string, number>;
}

export interface FrontendResponse {
  transcript: string;
  overall_score: number;
  speaking_rate: number;
  fluency_evaluator: {
    comment: string;
    fluency_score: number;
  };
  language_evaluator: {
    strengths: string[];
    improvements: string[];
    structure_score: number;
    grammar_score: number;
  };
  speech_evaluator: {
    strengths: string[];
    improvements: string[];
    clarity_score: number;
    confidence_score: number;
  };
  posture_evaluator: {
    tips: string[];
    posture_score: number;
  };
}
```
Use these to validate the payload you send to the service.
---