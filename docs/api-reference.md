# API Reference

This section describes the core functions, their signatures, inputs, outputs, and key behaviors across the backend modules.

---

## `fusion.get_feedback`

### Signature
```python
async def get_feedback(
    audio_path: str,
    fluency_model,
    posture_features: dict | PostureFeatures,
    response_schema = FrontendResponse,
    llm_model: str = "llama-3.3-70b-versatile",
) -> str
```

### Description

Runs the full feedback pipeline:

1. Extracts acoustic features from audio (`audio_utils.extract_features`).
2. Builds an LLM prompt (`prompt.get_prompt`).
3. Calls Groq LLM to generate structured feedback.
4. Post-processes results (WPM, overall score, transcript).
5. Returns a JSON string matching `FrontendResponse`.


### Parameters

- **audio_path** *(str)* — path to input audio file.  
- **fluency_model** — model handle passed to `extract_features`.  
- **posture_features** *(dict | PostureFeatures)* — frontend-provided posture signals.  
- **response_schema** *(Pydantic model)* — output schema (default: `FrontendResponse`).  
- **llm_model** *(str)* — Groq model ID (default: `"llama-3.3-70b-versatile"`).  


### Returns

JSON string with fields:

- **transcript** *(string)*  
- **speaking_rate** *(int, WPM)*  
- **overall_score** *(int, 0–100)*  
- **fluency_evaluator.comment** *(string)*  
- **fluency_evaluator.fluency_score** *(int, 0–100)*  
- **language_evaluator.\***  
- **speech_evaluator.\***  
- **posture_evaluator.\***

---

## audio_utils.extract_features

### Signature
```python
async def extract_features(
    audio_data: str | io.BytesIO,
    fluency_model,
    client: AsyncClient,
) -> dict
```
## Description
Asynchronously extracts acoustic and fluency features from audio input.  
Accepts either a file path or in-memory `BytesIO` object, processes with the provided fluency model, and returns low-level metrics for downstream scoring.

### Returns
A dictionary including:

- **speaking_rate** *(float, words per second → converted to integer WPM later)*
- **transcript** *(ASR string from model output)*
- **long_pause_count**, **long_pause_duration** *(seconds)*
- **baseline_\*** keys *(e.g., baseline_* speaking/energy/modulation stats)*
- **\*_delta** keys *(feature changes relative to baseline)*
- Additional model-derived metrics

---

## prompt.get_prompt

### Signature
```python
def get_prompt(
    audio_features: dict,
    posture_features: dict | PostureFeatures,
    response_schema=FrontendResponse,
) -> str
```

### Description
Builds the JSON-oriented LLM prompt.
Injects schema definition (FrontendResponse) so the LLM always returns valid JSON.

---

## utils.calculate_overall_score

### Signature
```python
def calculate_overall_score(result: dict) -> int
```

### Description

Computes a weighted composite score (0–100) from:

- **fluency_evaluator.fluency_score**
- **language_evaluator.structure_score**
- **language_evaluator.grammar_score**
- **speech_evaluator.clarity_score**
- **speech_evaluator.confidence_score**

---

## schema models

### PostureFeatures
```python
class PostureFeatures(BaseModel):
    eyeContact: Dict[str, float]
    shoulderAlignment: Dict[str, float]
    handGestures: Dict[str, float]
    headBodyAlignment: Dict[str, float]
```

### LLM Output Models
```python
class FluencyEvaluator(BaseModel):
    comment: str
    score: int  # Not used; see FrontendResponse, which expects `fluency_score`

class ContentEvaluator(BaseModel):
    strengths: List[str]
    improvements: List[str]
    structure_score: int
    grammar_score: int

class SpeechEvaluator(BaseModel):
    strengths: List[str]
    improvements: List[str]
    clarity_score: int
    confidence_score: int

class PostureEvaluator(BaseModel):
    tips: List[str]
    posture_score: int
```

### FrontendResponse (what the service returns after post-processing)
```python
class FrontendResponse(BaseModel):
    transcript: str
    overall_score: int
    speaking_rate: int
    fluency_evaluator: FluencyEvaluator  # In practice the service fills `fluency_score`
    language_evaluator: ContentEvaluator
    speech_evaluator: SpeechEvaluator
    posture_evaluator: PostureEvaluator
```

---

## Environment
- **Groq API**: Requires `groq` Python SDK and `GROQ_API_KEY`.
- Tested with **Python 3.9–3.10**.
- **Dependencies**: `librosa`, `parselmouth`, `pydub`, `soundfile`, `nltk`, `numpy`, `pydantic`, `joblib`, `groq`.