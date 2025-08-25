# Audio Analysis (Overview)

This document explains the **role of audio in TruthSense** — why it matters and what it contributes to the system.  It describes how audio features such as fluency, prosody, and rhythm are extracted and then fused with video/posture analysis to form holistic feedback.

The audio pipeline in TruthSense extracts meaningful acoustic and linguistic features from the candidate’s recorded response. These features are later fused with video features to generate a holistic feedback report.

## Workflow

1. **Audio Extraction**  
   - The uploaded interview video is split into a separate audio track using tools like **FFmpeg** or directly via **Librosa**.

2. **Feature Engineering**  
   - **Low-level descriptors**: pitch, energy, zero crossing rate, short-term spectral features.  
   - **MFCCs**: capture timbre and articulation of speech.  
   - **Delta MFCCs**: capture temporal variations and dynamics.  
   - **Speaking rate & pauses**: derived from silence detection and energy envelopes.  
   - **Prosodic features**: rhythm, stress, and intonation patterns.  

3. **Modeling**  
   - Extracted features are passed into ML models trained to evaluate delivery quality:  
     - **Random Forest / XGBoost** for baseline scoring and interpretability.  
     - **CNN-based models** on spectrograms for richer acoustic feature learning.  
     - **Hybrid approaches** combining handcrafted features + learned embeddings.  

4. **Output**  
   - The system generates structured JSON with:  
     - Fluency score  
     - Confidence level  
     - Filler-word detection  
     - Voice modulation / pitch variation  
     - Speaking pace and pauses  

These outputs are then **fused with video-based features** before being sent to the LLM for higher-level interpretation.

---

## Visual Workflow

``` mermaid
flowchart TD
    A[Video Upload] --> B[Audio Track Extraction]
    B --> C["Low-level Features (Pitch, Energy, ZCR)"]
    B --> D["MFCCs and Delta MFCCs"]
    B --> E["Speaking Rate & Pauses"]
    B --> F["Prosodic Features"]
    
    C --> G[Feature Set]
    D --> G
    E --> G
    F --> G
    
    G --> H["ML Models (RF / XGBoost / CNN)"]
    H --> I["Fusion with Video Features"]
    I --> J["LLM for Final Report"]
    J --> K["JSON Output (Fluency, Confidence, Pace, Modulation)"]
```

---

## Notes for Developers

- **Dependencies**: `librosa`, `numpy`, `scipy`, and `scikit-learn` (or PyTorch for deep models).  
- **File handling**: Audio paths are system-dependent. Ensure the pipeline’s init function points to the correct local paths.  
- **Calibration**: Thresholds for silence/energy may vary by microphone, recording environment, and background noise.  
- **Extensibility**: Additional prosodic features (intonation contours, stress patterns) can be integrated with minimal code changes.  
- **Performance**: Pre-cache MFCCs for longer videos to avoid recomputation during experimentation.  

## Implementation Notes

- **Async transcription**: Audio is split into chunks and transcribed in parallel for efficiency.  
- **Syllable rate**: Uses CMU Pronouncing Dictionary to estimate syllable counts from transcripts.  
- **Voice quality**: Parselmouth provides jitter, shimmer, and harmonicity features.  
- **Short clips**: If audio < 160 frames, pipeline exits early (insufficient data).  
- **Experimental functions**: `extract_from_wave` and some async extractors are not fully tested and may be revised.  