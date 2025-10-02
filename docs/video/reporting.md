# Reporting Module

The **Reporting Module** consolidates analysis outputs from **eye-contact**, **posture**, and **gesture** analysis into structured feedback reports. It is designed to generate both **real-time session summaries** and **post-session reports** for users.

---

## Features
- Aggregates results from all analysis modules.  
- Tracks frame-by-frame feedback counts.  
- Provides summary statistics at the end of a session.  
- Supports optional saving to disk in JSON format.  
- Flexible output: console print, file save, or structured return.

---

## Workflow

1. **During Session**  
   - Each frame’s analysis results update the `feedback_counts` dictionary.  
   - Categories include:  
     - `eyeContact`  
     - `shoulderAlignment`  
     - `headBodyAlignment`  
     - `hands`  

2. **At Session End**  
   - Results are collated into a final summary.  
   - Optionally displayed as a formatted report in the console.  
   - Optionally saved to a file if a `save_location` is provided.

---

## Example Output

### Console Report

**Eye Contact:**
- Maintained: 250 frames  
- Off-center: 45 frames  
- Blinking: 30 frames  

**Posture:**
- Good shoulder alignment: 180 frames  
- Slight shoulder tilt: 120 frames  
- Shoulders tilted: 25 frames  

**Head Alignment:**
- Centered: 260 frames  
- Slight tilt: 50 frames  
- Tilted: 15 frames  

**Gestures:**
- One hand high: 30 frames  
- Both hands visible: 90 frames  
- Hands not in frame: 205 frames  

### JSON Output
```json
{
  "eyeContact": {"Maintained": 250, "Off-center": 45, "Blinking": 30},
  "shoulderAlignment": {"Good": 180, "Slight tilt": 120, "Tilted": 25},
  "headBodyAlignment": {"Centered": 260, "Slight tilt": 50, "Tilted": 15},
  "hands": {"One high": 30, "Both visible": 90, "Not in frame": 205}
}
```
## Parameters

| Parameter      | Default | Description                                         |
|----------------|---------|-----------------------------------------------------|
| save_location  | None    | File path for saving final report (JSON format).    |
| display        | True    | Whether to display the final summary in console.    |

## Usage

```python
from body_language_corrector import BodyLanguageCorrector

blc = BodyLanguageCorrector()
blc.run_video_analysis(
    video_source=0,
    show_viz=False,
    display=True,
    save_location="session_report.json"
)
```
## Notes

- Reports are **summarized by frame counts**, not duration.  
- JSON format ensures **machine-readability** for further analysis.  
- Console summary is optimized for **human readability**.  

---
