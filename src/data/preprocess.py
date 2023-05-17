import os
import numpy as np
from sklearn.model_selection import train_test_split
from mfcc_extraction import parse_audio_files

def load_data(dataset_dir):
    # Parse audio files and extract features
    sub_dirs = [""] # Use empty string to include all subdirectories
    features, labels = parse_audio_files(dataset_dir, sub_dirs)

    # Load labels
    labels = np.load(os.path.join(dataset_dir, "labels.npy"))

    # Split data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    return x_train, x_val, y_train, y_val


def preprocess_data(x_train, x_val):
    # Normalize features
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train_norm = (x_train - mean) / std
    x_val_norm = (x_val - mean) / std

    return x_train_norm, x_val_norm

