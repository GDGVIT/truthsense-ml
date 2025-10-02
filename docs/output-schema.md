# Output Schema

The TruthSense feedback pipeline returns a **standardized JSON object** to the frontend.  
This schema integrates **audio fluency, language, speech clarity, and posture evaluation**, ensuring consistency across:

- **Frontend UI** (visualizations, dashboards)  
- **LLM prompting** (structured feedback requests)  
- **Downstream analysis** (training, benchmarking, improvement tracking)  

---

## High-Level Structure

```python
class FrontendResponse(BaseModel):
    transcript: str
    overall_score: int
    speaking_rate: int
    fluency_evaluator: FluencyEvaluator
    language_evaluator: ContentEvaluator
    speech_evaluator: SpeechEvaluator
    posture_evaluator: PostureEvaluator
```

---

## Field Definitions

### transcript *(string)*
Full transcription of the input audio.  
- Drives text display in the frontend.  
- Serves as context for LLM-based feedback.  

---

### overall_score *(int, 0–100)*
Composite score aggregating all evaluators:  
- Fluency  
- Language  
- Speech  
- Posture  

---

### speaking_rate *(int, WPM)*
Words per minute, derived from transcript length and audio duration.  

---

## Nested Evaluators

### fluency_evaluator *(object → FluencyEvaluator)*
Captures delivery aspects like flow, pauses, and fillers.  
```python
class FluencyEvaluator(BaseModel):
    comment: str  # 3–5 sentence summary
    score: int    # 0–100
```
- **comment**:  Natural-language feedback on pacing, fillers, pauses, rhythm.
- **score**:  Normalized fluency score (0–100).

---

### language_evaluator *(object → ContentEvaluator)*
Analyzes content structure and grammar.
```python
class ContentEvaluator(BaseModel):
    strengths: List[str]       # 2–5 key strengths
    improvements: List[str]    # 2–5 improvement points
    structure_score: int       # 0–100
    grammar_score: int         # 0–100
```
- **strengths**: Concise positives about structure, argument clarity, or grammar.  
- **improvements**: Specific actionable feedback.  
- **structure_score**: Logical flow and organization *(0–100)*.  
- **grammar_score**: Accuracy of language use *(0–100)*.  

---

### speech_evaluator *(object → SpeechEvaluator)*
Assesses clarity and confidence in delivery.  
```python
class SpeechEvaluator(BaseModel):
    strengths: List[str]       # 2–5 clarity/confidence highlights
    improvements: List[str]    # 2–5 clarity/confidence suggestions
    clarity_score: int         # 0–100
    confidence_score: int      # 0–100
```
- **strengths**: Highlights strong aspects of articulation, tone, or delivery.  
- **improvements**: Actionable suggestions for improving clarity, projection, or confidence.  
- **clarity_score**: Score for understandability and articulation *(0–100)*.  
- **confidence_score**: Score for perceived speaker confidence *(0–100)*. 
--- 

### posture_evaluator *(object → PostureEvaluator)*
Encodes video-based analysis of body language.  
```python
class PostureEvaluator(BaseModel):
    tips: List[str]     # 3–7 posture improvement suggestions
    posture_score: int  # 0–100
```
- **tips**: Actionable suggestions for improving eye contact, gesture use, or alignment *(3–7 items)*.  
- **posture_score**: Score reflecting proper posture maintenance and body language effectiveness *(0–100)*.  
---

### Downstream Usage
- **Frontend UI**: Drives multiple dashboards (e.g., fluency bars, posture alignment overlays, speech clarity meters).  
- **LLM Prompting**: Injected into structured prompts for generating natural-language feedback.  
- **Training/Analytics**: Provides labeled fields to benchmark or fine-tune future scoring models.  
---
