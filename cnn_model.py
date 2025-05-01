import tensorflow as tf
from tensorflow.keras import models, layers

def build_model():
    model = models.Sequential()
    
    # 1st Convolution Layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 2nd Convolution Layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 3rd Convolution Layer
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Flatten Layer
    model.add(layers.Flatten())

    # Fully Connected Layer 1
    model.add(layers.Dense(256, activation='relu'))

    # Fully Connected Layer 2
    model.add(layers.Dense(128, activation='relu'))

    # Output Layer (10 classes, softmax for multi-class classification)
    model.add(layers.Dense(10, activation='softmax'))

    # Use categorical_crossentropy since labels are now one-hot encoded
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
