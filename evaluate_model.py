import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import numpy as np

# Load CIFAR-10 test data
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalize the test data
x_test = x_test / 255.0

# Convert labels to one-hot encoding
y_test_one_hot = to_categorical(y_test, 10)

# Load the trained model
model = tf.keras.models.load_model("cnn_model.h5")

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test_one_hot)

# Predict test labels
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert y_test to integer labels for classification report
y_test_int = y_test.flatten()

# Generate classification report
report = classification_report(y_test_int, y_pred, target_names=[
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
])

# Print results
print(f"\nCNN Model - Test Accuracy: {accuracy * 100:.2f}%")
print(f"CNN Model - Test Loss: {loss:.4f}")
print("\nClassification Report for CNN Model:")
print(report)

# Save report to a file
with open("classification_report.txt", "w") as file:
    file.write(f"CNN Model - Test Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"CNN Model - Test Loss: {loss:.4f}\n\n")
    file.write("Classification Report for CNN Model:\n")
    file.write(report)

print("\nClassification report saved as 'classification_report.txt'")
