import os
import librosa
import glob
import numpy as np

def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=22050)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)

    return mfccs_scaled

def parse_audio_files(parent_dir, sub_dirs, file_ext="*.wav"):
    features, labels = np.empty((0, 13)), np.empty(0)

    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs = extract_features(fn)
            features = np.vstack([features, mfccs])
            labels = np.append(labels, fn.split(os.sep)[-1].split('_')[1])

    return np.array(features), np.array(labels, dtype=np.int)

# Example usage
parent_dir = "dataset"
sub_dirs = ["raw"]  # Use empty string to include all subdirectories

features, labels = parse_audio_files(parent_dir, sub_dirs)
