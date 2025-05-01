import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load Model
model = load_model("cnn_model.h5")
print("✅ Model Loaded Successfully!")

# CIFAR-10 Classes
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load CIFAR-10 Dataset
(_, _), (x_test, y_test) = cifar10.load_data()
x_test = x_test.astype('float32') / 255.0  # Normalize Image

# Predict First 10 Images
predicted_labels = []
actual_labels = []

plt.figure(figsize=(12, 6))

# Predicted Images
for i in range(10):
    img = np.expand_dims(x_test[i], axis=0)
    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]
    actual_label = labels[y_test[i][0]]

    predicted_labels.append(predicted_label)
    actual_labels.append(actual_label)

    # Plot predicted images
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], interpolation='nearest')
    plt.title(f"Pred: {predicted_label}")
    plt.axis("off")

plt.suptitle("Predicted Images")
plt.savefig("predicted_images.png")  # ✅ Save predicted images
plt.show()

# Actual Images
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], interpolation='nearest')
    plt.title(f"Actual: {actual_labels[i]}")
    plt.axis("off")

plt.suptitle("Actual Images")
plt.savefig("actual_images.png")  # ✅ Save actual images
plt.show()

print("✅ Images saved as 'predicted_images.png' and 'actual_images.png'.")
