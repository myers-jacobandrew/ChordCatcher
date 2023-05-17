import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, x_test, y_test):
    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    # Generate predictions on test data
    y_pred = model.predict(x_test)

    # Convert predictions from one-hot encoding to label
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Print classification report and confusion matrix
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
