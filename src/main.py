import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from data.mfcc_extraction import extract_features
from data.preprocess import preprocess_data
from models.model_definition import create_model
from models.evaluate_model import evaluate_model


# Set your dataset directory path
dataset_dir = "\\ChordCatcher\\dataset\\raw"

# Extract MFCC features from dataset
features, labels = extract_features(dataset_dir)
num_classes = len(np.unique(labels))

# Preprocess data
x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
x_train_norm, x_val_norm = preprocess_data(x_train, x_val)

# Define model architecture
model = create_model(input_shape=x_train[0].shape, num_classes=num_classes)

# Compile and Train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train_norm, y_train, validation_data=(x_val_norm, y_val), epochs=10, batch_size=32)

# Prepare test data
x_test_norm, y_test = preprocess_data(x_val, y_val)

# Evaluate the model
evaluate_model(model, x_test_norm, y_test)


if __name__ == "__main__":
    # Run the main code
    pass
