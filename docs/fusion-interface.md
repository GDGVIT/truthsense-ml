# Fusion & LLM

This module **fuses acoustic features + posture features** and generates structured feedback using an LLM.

---

## Overview

`get_feedback(...)` orchestrates:

1. **Audio analysis** → `audio_utils.extract_features(...)`  
2. **Prompt building** → `prompt.get_prompt(...)` (injects the Pydantic JSON schema so the LLM returns valid JSON)  
3. **LLM call** → Groq `AsyncClient().chat.completions.create(...)`  
4. **Post-processing** → converts speaking rate to WPM, computes `overall_score`, attaches transcript  
5. **Returns** → a JSON **string** matching `schema.FrontendResponse` plus a few extra fields.

---

## Function Signature

```python
async def get_feedback(
    audio_path: str,
    fluency_model,  # model used by extract_features
    posture_features: dict | PostureFeatures,
    response_schema = FrontendResponse,
    llm_model: str = "llama-3.3-70b-versatile",
) -> str
```

## Input

- **audio_path** — path to the user’s audio file.  
- **fluency_model** — model handle consumed by `extract_features`.  
- **posture_features** — dict or `PostureFeatures` (see *Frontend Interface* doc).  
- **response_schema** — Pydantic model (default: `FrontendResponse`).  
- **llm_model** — Groq model ID (default: `llama-3.3-70b-versatile`).  

---

## Output

- JSON string that includes (at minimum):  
  - Fields defined by **FrontendResponse**  
  - `speaking_rate` (WPM, integer)  
  - `overall_score` (from `utils.calculate_overall_score`)  
  - `transcript` (from `audio_features['transcript']`)  

---

## Data Flow
```json
audio_path ─┐
└── extract_features(…) ──▶ audio_features
posture_features ───────────────────────▶

audio_features + posture_features + schema
──▶ get_prompt(…) ──▶ prompt (str)

prompt
──▶ Groq LLM ──▶ JSON (dict)
──▶ post-process (WPM / overall_score / transcript)
──▶ JSON string
```
### Note:
Posture analysis has been implemented in-client, and so the backend folder doesn't include that.

---

## LLM Call (key settings)

- `temperature=0.5, top_p=1`
- `max_completion_tokens=32768`
- `response_format={"type": "json_object"}`
- `stream=False`

---

## Example (Python)

```python
import asyncio
from fusion import get_feedback
from schema import PostureFeatures, FrontendResponse

async def main():
    # posture features may also be plain dict — see Frontend docs
    posture = PostureFeatures(...)
    feedback_json = await get_feedback(
        audio_path="sample.wav",
        fluency_model=...,
        posture_features=posture,
        response_schema=FrontendResponse,
        llm_model="llama-3.3-70b-versatile",
    )
    print(feedback_json)

asyncio.run(main())
```

---

## Environment / Dependencies

- **Groq**: `groq` Python SDK; requires `GROQ_API_KEY` in environment.
- `audio_utils.extract_features(...)`
- `prompt.get_prompt(...)`
- `utils.calculate_overall_score(...)`
- `schema.PostureFeatures`, `schema.FrontendResponse` (Pydantic).

---

## Post-processing Details

- `response['speaking_rate'] = int(audio_features['speaking_rate'] * 60)`  
  (Assumes `speaking_rate` was per-second in `audio_features`.)

- `response['overall_score'] = calculate_overall_score(response)`

- `response['transcript'] = audio_features['transcript']`

---

For detailed schema definitions, feature keys, and scoring weights, see [API Reference](./api-reference.md).