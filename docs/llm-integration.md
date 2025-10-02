# LLM Integration

This module handles the **prompt construction and interaction with the LLM**. It ensures that acoustic features, posture features, and schema constraints are injected into a structured prompt, so that the LLM always returns valid JSON feedback.

---

## Overview

1. **Format posture features** → Convert `PostureFeatures` (Pydantic or dict) into a readable Markdown-like section.  
2. **Assemble full prompt** → Combine audio metrics, relative changes, transcript, and posture summary into a structured instruction block.  
3. **Inject schema** → The target Pydantic model (`FrontendResponse`) is serialized to JSON Schema and appended, forcing the LLM to comply.  
4. **LLM call** → Used inside `fusion.get_feedback(...)` with Groq `AsyncClient`.  

---

## Key Functions

### `format_posture_features(posture_features: dict | PostureFeatures) -> str`

Converts posture feature inputs into human-readable Markdown text for embedding into the prompt.  

- **Inputs**:  
  - `dict` or `PostureFeatures` instance with sections:  
    - `eyeContact`  
    - `shoulderAlignment`  
    - `headBodyAlignment`  
    - `handGestures`  

- **Output**:  
  - String like:

    ```text
    ## Eye Contact
    - left: 0.8
    - right: 0.2
    ```

---

### `get_prompt(audio_features: dict, posture_features: dict | PostureFeatures, response_schema=FrontendResponse) -> str`

Builds the full system prompt that guides the LLM.  

- **Inputs**:  
  - `audio_features` — dict returned from `audio_utils.extract_features`.  
  - `posture_features` — dict or `PostureFeatures`.  
  - `response_schema` — Pydantic model (default: `FrontendResponse`).  

- **Output**:  
  - A **long, structured prompt string** containing:  
    - Transcript  
    - Baseline metrics (first N seconds)  
    - Full-clip metrics  
    - Relative changes from baseline  
    - Posture features  
    - Interpretation tips  
    - Instructions for 4 evaluators (Fluency, Language, Speech, Posture)  
    - JSON output schema  

---

## Output Contract

The prompt ends by enforcing this JSON schema:

```json
{
  "fluency_evaluator": {
    "comment": "string",
    "fluency_score": "int"
  },
  "language_evaluator": {
    "strengths": ["string"],
    "improvements": ["string"],
    "structure_score": "int",
    "grammar_score": "int"
  },
  "speech_evaluator": {
    "strengths": ["string"],
    "improvements": ["string"],
    "clarity_score": "int",
    "confidence_score": "int"
  },
  "posture_evaluator": {
    "tips": ["string"],
    "score": "int"
  }
}
```
This ensures all downstream consumers (frontend, scoring) receive predictable JSON.

---

## Integration Point

- `fusion.get_feedback(...)` calls `get_prompt(...)` before invoking the LLM.  
- Groq LLM is configured with `response_format={"type": "json_object"}`, so the response parses directly into JSON.  

---

## Dependencies

- `schema.PostureFeatures`, `schema.FrontendResponse` (Pydantic)  
- `audio_utils.extract_features` (for feature keys injected into the prompt)  
- `groq.AsyncClient`  
---