# Output Schema (Audio)

The audio analysis pipeline in TruthSense always returns a **standardized JSON object**.  
This schema ensures consistency across the frontend, LLM prompting, and downstream analysis.

---

## High-Level Structure

```json
{
  "transcript": "string",
  "speaking_rate": "float",
  "syllable_rate": "float",
  "fluency_scores": {
    "overall": "float",
    "filler_words": "float",
    "pauses": "float",
    "intonation": "float"
  },
  "pause_statistics": {
    "total_pauses": "int",
    "avg_pause_duration": "float",
    "longest_pause": "float"
  },
  "performance_score": "float"
}
```

---
## Field Definitions

- **`transcript`** *(string)*  
  Continuous text transcript of the audio.

- **`speaking_rate`** *(float)*  
  Words per second, calculated from transcript length and audio duration.

- **`syllable_rate`** *(float)*  
  Average syllables per second, computed via CMUdict-based syllable counts.

- **`fluency_scores`** *(object)*  
  - `overall`: Unified fluency score (0–1).  
  - `filler_words`: Proportion of fillers detected.  
  - `pauses`: Fluency loss from silence frequency/duration.  
  - `intonation`: Pitch and modulation smoothness.  

- **`pause_statistics`** *(object)*  
  - `total_pauses`: Number of pauses detected.  
  - `avg_pause_duration`: Mean silence length (seconds).  
  - `longest_pause`: Longest silence span (seconds).  

- **`performance_score`** *(float)*  
  Final normalized score (0–100) combining all metrics.

---

## Downstream Usage

- **Frontend UI**: Drives visualizations (pause charts, pitch plots, fluency bars).
- **LLM Prompting**: Injected into structured prompt for contextual feedback.
- **Model Training**: Labels/targets can be aligned with schema fields.

---
