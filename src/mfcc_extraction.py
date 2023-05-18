import os
import librosa
import glob
import numpy as np

def extract_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        return mfccs_scaled
    except PermissionError as e:
        print("Permission denied:", e)
        return None

def parse_audio_files(directory):
    try:
        file_list = []
        label_list = []
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                label = filename.split("_")[2]  # Extract label after the second underscore
                file_list.append(os.path.join(directory, filename))
                label_list.append(label)
        return file_list, label_list
    except PermissionError as e:
        print("Permission denied:", e)
        return [], []

directory = "dataset/raw/"
file_list, label_list = parse_audio_files(directory)

if file_list is not None:
    features = []
    labels = []
    for file_path in file_list:
        feature = extract_features(file_path)
        if feature is not None:
            features.append(feature)
            labels.append(label_list[file_list.index(file_path)])
    
