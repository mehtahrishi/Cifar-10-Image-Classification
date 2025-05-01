import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalize the test images
x_test = x_test / 255.0

# Class names for CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load the trained model
model = tf.keras.models.load_model("cnn_model_v2.h5")

# Get predictions for the first 10 test images
predictions = model.predict(x_test[:10])
predicted_labels = np.argmax(predictions, axis=1)

# **1st Image: Actual Labels**
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Actual: {class_names[y_test[i][0]]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("actual_images_v2.png")  # Save actual images
plt.show()
print("Actual images saved as 'actual_images_v2.png'.")

# **2nd Image: Predicted Labels**
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Predicted: {class_names[predicted_labels[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("predicted_images-v2.png")  # Save predicted images
plt.show()
print("Predicted images saved as 'predicted_images_v2.png'.")
