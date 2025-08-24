# Fluency Model

## Overview
The **Fluency Model** is a supervised machine learning classifier trained on acoustic and prosodic features extracted from speech audio. Its primary role is to predict the speaker’s **fluency level** (e.g., *Low*, *Medium*, *High*) by comparing baseline and full-clip features.

It forms a core part of the audio analysis pipeline, enabling automatic scoring of fluency in spoken delivery.

---

## Data
- **Inputs:** Balanced dataset of acoustic/prosodic features (`X.npy`) and fluency labels (`y.npy`).
- **Features used:**
  - Zero Crossing Rate (ZCR)  
  - Pitch statistics (mean, std, variance)  
  - Root Mean Square (RMS) energy (mean, std, var)  
  - Mel-Frequency Cepstral Coefficients (MFCCs) and ΔMFCC  
  - Jitter, shimmer, and Harmonic-to-Noise Ratio (HNR)  
  - Speaking rate, syllable rate, and pause metrics  

- **Labels:** 3-class fluency scale:
  - `0` → Low  
  - `1` → Medium  
  - `2` → High  

---

## Model
The baseline fluency model is implemented as a **Random Forest Classifier**:

- `n_estimators = 200`  
- `max_depth = 10`  
- `random_state = 42`  

This model was chosen for its interpretability, strong performance on tabular feature sets, and resilience against overfitting on small to medium-sized datasets.

---

## Training & Evaluation
- **Split:** 80% training / 20% testing, stratified by class.  
- **Evaluation Metric:** `classification_report` (precision, recall, F1-score, support).  
- **Cross-validation:** 5-fold CV used during tuning.  
- **Performance:** Accuracy ~85–90% on validation depending on feature set.  

## Outputs

- **Predictions:** Stored in `fluency/outputs/fluency_preds.npy`  
- **True labels:** Stored in `fluency/outputs/y_test.npy`  
- **Trained model weights:** Saved at `fluency/weights/fluency_model.pkl`  

---

## Usage

### Load Model
```python
import joblib
model = joblib.load("fluency/weights/fluency_model.pkl")
import numpy as np
X_sample = np.array([...])  # feature vector
pred = model.predict([X_sample])
print(pred)  # -> 0 (Low), 1 (Medium), or 2 (High)
```