import os
import numpy as np
import librosa
from tqdm import tqdm
import parselmouth
from parselmouth.praat import call
import warnings
warnings.filterwarnings("ignore")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        if y.shape[0] < sr // 2:  # skip if too short
            return None

        # ZCR
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        sound = parselmouth.Sound(values=y, sampling_frequency=sr)
        
        pitch_values = sound.to_pitch().selected_array['frequency']
        pitch_mean = np.nanmean(pitch_values)
        pitch_std = np.nanstd(pitch_values)

        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_var = np.var(rms)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(mfcc_delta)

        return [
            zcr, pitch_mean, pitch_std,
            rms_mean, rms_std, rms_var,
            mfcc_mean, delta_mean
        ]
    except Exception as e:
        print(f"[SKIPPED] {file_path}: {e}")
        return None


def load_dataset(dataset_path):
    labels = {'001 - Low': 0, '002 - Intermediate': 1, '003 - High': 2}
    X, y = [], []

    for label_folder in labels:
        folder_path = os.path.join(dataset_path, label_folder)
        label = labels[label_folder]
        print(f"Processing {label_folder}:")
        
        files = [f for f in os.listdir(folder_path)]
        for f in tqdm(files):
            full_path = os.path.join(folder_path, f)
            features = extract_features(full_path)
            if features is not None and not np.isnan(features).any():
                X.append(features)
                y.append(label)
            else:
                print(f"[SKIPPED] Invalid or NaN in {f}")

    return np.array(X), np.array(y)


if __name__ == "__main__":
    X, y = load_dataset("fluency/data/")
    print(f"Final shape of X: {X.shape}, y: {y.shape}")
    os.makedirs('fluency/outputs', exist_ok=True)
    np.save("fluency/outputs/X.npy", X)
    np.save("fluency/outputs/y.npy", y)