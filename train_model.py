from dataset_preprocessing import load_preprocess_data, get_augmented_data
from cnn_model import build_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load & Preprocess Data
x_train, y_train, x_test, y_test = load_preprocess_data()

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print("New y_train shape:", y_train.shape)  # Should be (50000, 10)
print("New y_test shape:", y_test.shape)  # Should be (10000, 10)

# Apply Data Augmentation
datagen = get_augmented_data(x_train)

# Build Model
model = build_model()

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model using Augmented Data
history = model.fit(datagen.flow(x_train, y_train, batch_size=64), 
                    epochs=10, 
                    validation_data=(x_test, y_test))

# Save Model
model.save("cnn_model.h5")
print("✅ Model Saved Successfully!")

# Plot Accuracy Graph
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot Loss Graph
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Evaluate Model on Test Data
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Compute Classification Metrics
print("📊 Classification Report:")
print(classification_report(y_true_classes, y_pred_classes))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

import matplotlib.pyplot as plt

# Save Accuracy Graph
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy Graph')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.savefig("accuracy_graph.png")  # ✅ Ensure the file is saved

# Save Loss Graph
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss Graph')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("loss_graph.png")  # ✅ Ensure the file is saved

plt.show()
