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

import os

def parse_audio_files(directory):
    file_list = []
    label_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            label = filename.split("_")[2]  # Extract label after the second underscore
            file_list.append(os.path.join(directory, filename))
            label_list.append(label)
    return file_list, label_list

'''
parent_dir = "dataset"
sub_dirs = ["raw"]  # Use empty string to include all subdirectories
'''
directory = "dataset/raw/"
file_list, label_list = parse_audio_files(directory)

