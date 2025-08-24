# TruthSense

TruthSense is an AI-powered tool designed to help candidates **practice and improve interview performance**.  
By analyzing both the **audio** and **video** streams of a recorded response, it generates detailed feedback on verbal fluency, confidence, body language, and overall delivery.

---

## Why TruthSense?

- **Interviews are stressful.** Candidates often fail to notice their own nervous habits, filler words, or weak posture.  
- **Traditional mock interviews are limited.** Human feedback can be subjective, inconsistent, or unavailable on demand.  
- **AI assistance provides objectivity.** TruthSense combines signal processing, computer vision, and large language models to provide structured, actionable insights.

---

## Who Is It For?

- **Students and job seekers** preparing for placement interviews.  
- **Professionals** practicing presentations or public speaking.  
- **Educators and trainers** who want scalable tools for communication coaching.

---

## System Overview

TruthSense follows a modular pipeline:

1. **Input**: User uploads a recorded interview practice video.  
2. **Audio Analysis**: Extracts MFCCs, pitch, speaking rate, pauses, and prosody features.  
3. **Video Analysis**: Uses Mediapipe and OpenCV for gaze tracking, blink detection, posture, and gestures.  
4. **Feature Fusion**: Combines audio and video insights.  
5. **LLM Integration**: Passes extracted features to an LLM to generate a coherent, human-readable feedback report.  
6. **Output**: A structured JSON report rendered in the frontend as easy-to-read feedback.

For a deeper dive, see [System Architecture](architecture.md), [Audio Analysis](audio/overview.md), and [Video Analysis](video/overview.md).

---

## Goals

- Provide **objective, repeatable feedback** for interview preparation.  
- Encourage **self-reflection** by highlighting patterns in verbal and non-verbal delivery.  
- Offer a **scalable platform** that can be integrated into training programs or used individually.