import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_preprocess_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize images (0-255 to 0-1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return x_train, y_train, x_test, y_test

def get_augmented_data(x_train):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(x_train)
    return datagen
