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

# Load the customized model
model = tf.keras.models.load_model("cnn_model_v2.h5")  # Or "cnn_model.h5" if you prefer

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test_one_hot)

# Predict test labels
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Generate classification report
report = classification_report(y_test, y_pred, target_names=[
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
])

print(f"\nCustomized Model - Test Accuracy: {accuracy * 100:.2f}%")
print(f"Customized Model - Test Loss: {loss:.4f}")
print("\nClassification Report for Customized Model:")
print(report)

# Save report to a file
with open("report_v2.txt", "w") as file:
    file.write(f"Customized Model - Test Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"Customized Model - Test Loss: {loss:.4f}\n\n")
    file.write("Classification Report for Customized Model:\n")
    file.write(report)
