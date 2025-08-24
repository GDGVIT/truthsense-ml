# Audio Pipeline (Implementation)

This document explains **how the audio analysis is implemented in code** — the step-by-step process from raw audio input to structured JSON output.  
It covers transcription, feature extraction, fluency modeling, and response packaging as part of the implementation.

---

## High-Level Flow

1. **Input Acquisition**  
   Accepts either an audio file or an in-memory buffer from the frontend.

2. **Transcription**  
   - Splits audio into manageable chunks using `pydub`.  
   - Performs asynchronous transcription on all chunks concurrently.  
   - Collates partial transcripts into a single structured object.  
   → See [Transcription](transcription.md)

3. **Baseline Definition**  
   - The first *N* seconds of audio are isolated.  
   - Baseline metrics (pitch, energy, fluency, syllable rate, etc.) are computed.  
   - All subsequent metrics are interpreted **relative to this baseline**.  

4. **Feature Extraction**  
   - Low-level features: ZCR, RMS energy, MFCCs, pitch contour.  
   - Higher-order metrics: speaking rate, syllable rate, jitter, shimmer, HNR.  
   - Uses CMU Pronouncing Dictionary for syllable counts.  
   → See [Feature Engineering](feature-engineering.md)

5. **Fluency Model Inference**  
   - Extracted features are passed into a trained ML model (XGBoost, Random Forest, or CNN-spectrogram).  
   - Model outputs confidence scores on fluency, filler usage, modulation, and overall delivery quality.  
   → See [Fluency Model](fluency-model.md)

6. **Response Packaging**  
   - Results are assembled into a standardized [JSON schema](output-schema.md).  
   - Includes transcript, fluency ratings, speaking rate, pauses, and performance scores.  
   - This object is returned to the frontend for visualization.  

---

## Core Function: `get_feedback`

The **`get_feedback`** function orchestrates the entire audio pipeline (implemented in `analyse.py` at the repository root):

```python
async def get_feedback(audio_path, fluency_model, posture_features, ...):
    audio_features = await extract_features(audio_path, fluency_model, llm_client)
    prompt = get_prompt(audio_features, posture_features, response_schema)
    completion = await llm_client.chat.completions.create(...)
    return json.dumps(response, indent=2)
```

### Inputs
- Audio file or buffer  
- Fluency model  
- Dictionary of posture features (from frontend)  
- Async LLM client (for feedback generation)  

---

### Steps
1. Transcribes and extracts audio features.  
2. Constructs a multimodal prompt (audio + posture).  
3. Calls the LLM to generate structured feedback.  

---

### Outputs
- JSON object containing:  
  - Transcript  
  - Speaking rate & syllable rate  
  - Fluency scores  
  - Pause statistics  
  - Overall performance score  
- See [Output Schema](output-schema.md) for the full contract.  

---

## Supporting Functions

- **`transcribe_audio`**  
  Async chunking with `pydub`, concurrent transcription, aggregation.  

- **`extract_features`**  
  Audio decoding with `soundfile`, low-level descriptors (ZCR, RMS, pitch, MFCCs), and high-level metrics (jitter, shimmer, HNR).  

- **`estimate_syllable_rate`**  
  Transcript-based syllable counts via CMUdict, converted into syllable rate.  

---

## End-to-End Behavior

The pipeline guarantees that:  

- Short clips are filtered out (minimum 160 frames).  
- Long recordings are segmented, transcribed, and reconstructed seamlessly.  
- Baseline metrics (first *N* seconds, default: 10s) are always included for relative comparison.  
- Outputs follow the fixed JSON schema expected by the frontend.  
## Visual Overview

``` mermaid
flowchart TD
    A[Audio Input] --> B[Transcription]
    B --> C[Baseline Definition]
    C --> D[Feature Extraction]
    D --> E[Fluency Model Inference]
    E --> F[LLM Feedback Generation]
    F --> G[JSON Response to Frontend]
```