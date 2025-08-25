<p align="center">
<a href="https://dscvit.com">
	<img width="400" src="https://user-images.githubusercontent.com/56252312/159312411-58410727-3933-4224-b43e-4e9b627838a3.png#gh-light-mode-only" alt="GDSC VIT"/>
</a>
	<h2 align="center"> < TruthSense > </h2>
	<h4 align="center"> < A multimodal AI system for evaluating communication skills using audio, video, and LLM-based feedback. > <h4>
</p>

---
[![Join Us](https://img.shields.io/badge/Join%20Us-Developer%20Student%20Clubs-red)](https://dsc.community.dev/vellore-institute-of-technology/)
[![Discord Chat](https://img.shields.io/discord/760928671698649098.svg)](https://discord.gg/498KVdSKWR)

[![DOCS](https://img.shields.io/badge/Documentation-see%20docs-green?style=flat-square&logo=appveyor)](INSERT_LINK_FOR_DOCS_HERE) 
  [![UI ](https://img.shields.io/badge/User%20Interface-Link%20to%20UI-orange?style=flat-square&logo=appveyor)](INSERT_UI_LINK_HERE)


## Features
- [x] Audio analysis (transcription, speaking rate, pauses, fluency metrics)  
- [x] Video analysis (eye contact, posture, gesture, alignment)  
- [x] Fusion layer combining multimodal features  
- [x] LLM-powered structured feedback (via Groq API)  
- [x] Validated output schema for frontend integration 

<br>

## Dependencies
- Python 3.9–3.10  
- [Groq API](https://groq.com/) SDK (requires `GROQ_API_KEY`)  
- Core libraries: `librosa`, `parselmouth`, `pydub`, `soundfile`, `nltk`, `numpy`, `pydantic`, `joblib`, `groq`, `mediapipe`, `opencv-python`

---


## Running

<directions to install>

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/truthsense-ml.git
cd truthsense-ml
```
### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```


<directions to execute>

### 4. Execute
```bash
from fusion import get_feedback
from schema import PostureFeatures

posture = PostureFeatures(
    eyeContact={"score": 0.8},
    shoulderAlignment={"score": 0.9},
    handGestures={"score": 0.7},
    headBodyAlignment={"score": 0.85},
)

feedback = await get_feedback(
    audio_path="sample.wav",
    fluency_model=my_fluency_model,
    posture_features=posture
)

print(feedback)
```

## Contributors

<table>
	<tr align="center">
		<td>
		John Doe
		<p align="center">
			<img src = "https://dscvit.com/images/dsc-logo-square.svg" width="150" height="150" alt="Your Name Here (Insert Your Image Link In Src">
		</p>
			<p align="center">
				<a href = "https://github.com/person1">
					<img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36" alt="GitHub"/>
				</a>
				<a href = "https://www.linkedin.com/in/person1">
					<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36" alt="LinkedIn"/>
				</a>
			</p>
		</td>
	</tr>
</table>

<p align="center">
	Made with ❤ by <a href="https://dscvit.com">GDSC-VIT</a>
</p>
