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
- [ ] Add Facial Emotion Recognition to improve facial analysis
- [ ] Add Speech Emotion Recognition to improve speech analysis and match transcript with emotion.

<br>

## Dependencies

- Python 3.9–3.10  
- [Groq API](https://groq.com/) SDK (requires `GROQ_API_KEY`)  
- Core libraries: `librosa`, `parselmouth`, `pydub`, `soundfile`, `nltk`, `numpy`, `pydantic`, `joblib`, `groq`, `mediapipe`, `opencv-python`, `fastapi`, `scikit-learn`, `xgboost`, `lightgbm`

---


## Running

There are two ways to run the server: using Docker (recommended) or running it locally in a Python environment.

### Running with Docker (Recommended)

This is the easiest and most reliable way to get the server running.

1. **Prerequisites**: Make sure you have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.

2. **Configure Environment**:
    In the root of the project, you will find a file named `.env.docker`. Open this file and replace the placeholder with your actual Groq API key:
   	```
    GROQ_API_KEY="<YOUR_GROQ_API_KEY>"
    ```

3. **Build and Run**:
    Open your terminal and run the following command from the project root:
    ```bash
    docker compose build && docker compose up
    ```
    The server will start, and you can access it at `http://localhost:8003`.

### Running Locally

1. **Clone the repository**:
    ```bash
    git clone https://github.com/GadiMahi/truthsense-ml.git
    cd truthsense-ml
    ```
2. **Create a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate   # macOS/Linux
    .venv\Scripts\activate      # Windows
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Configure Environment**:
    Create a file named `.env.local` and add your Groq API key:
    ```
    GROQ_API_KEY="<YOUR_API_KEY>"
    ```
5. **Run the server**:
    ```bash
    uvicorn backend.server:app --host 0.0.0.0 --port 8003 --env-file .env.local
    ```

## Contributors

<table>
	<tr align="center">
		<td>
			Mahi Gadi
			<p align="center">
				<img src="https://dscvit.com/images/dsc-logo-square.svg" width="150" height="150" alt="Mahi Gadi">
			</p>
			<p align="center">
				<a href="https://github.com/GadiMahi">
					<img src="http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height="36" alt="GitHub"/>
				</a>
				<a href="https://www.linkedin.com/in/mahigadi">
					<img src="http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36" alt="LinkedIn"/>
				</a>
			</p>
		</td>
		<td>
			Utkarsh Malaiya
			<p align="center">
				<img src="https://dscvit.com/images/dsc-logo-square.svg" width="150" height="150" alt="Utkarsh Malaiya">
			</p>
			<p align="center">
				<a href="https://github.com/utkrshm">
					<img src="http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height="36" alt="GitHub"/>
				</a>
				<a href="https://www.linkedin.com/in/utkarsh-malaiya">
					<img src="http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36" alt="LinkedIn"/>
				</a>
			</p>
		</td>
	</tr>
</table>

<p align="center">
	Made with ❤ by <a href="https://dscvit.com">GDSC-VIT</a>
</p>
