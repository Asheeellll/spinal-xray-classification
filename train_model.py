import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set image size and directories
IMG_SIZE = 224  # Image size (224x224)
TRAIN_DIR = 'C:/Users/Asheel/Desktop/mini project/Structure/training'
VAL_DIR = 'C:/Users/Asheel/Desktop/mini project/Structure/validation'

# Step 1: Load and Preprocess Data
def load_data(directory):
    data = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                data.append(image)
                labels.append(label)  # You can encode labels numerically if needed
    return np.array(data), np.array(labels)

# Step 2: Preprocess the images and labels
train_data, train_labels = load_data(TRAIN_DIR)
val_data, val_labels = load_data(VAL_DIR)

# Normalize images
train_data = train_data.astype('float32') / 255.0
val_data = val_data.astype('float32') / 255.0

# Encode labels (you can use LabelEncoder or one-hot encoding)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)

# Step 3: Create Data Generators with Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_data, train_labels, batch_size=32)
val_generator = val_datagen.flow(val_data, val_labels, batch_size=32)

# Step 4: Build the Model (Simple CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer for multi-class
])

# Step 5: Compile the Model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(train_generator, epochs=10, validation_data=val_generator)

# Step 7: Save the Model
model.save('spinal_xray_model.h5')

# Step 8: Plot Training History
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)

