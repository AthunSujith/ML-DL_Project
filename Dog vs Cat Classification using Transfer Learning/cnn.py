# ✅ Step 1: Install necessary libraries


# ✅ Step 2: Imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

import os
import zipfile
import pathlib

# ✅ Step 3: Download cat vs dog dataset from TensorFlow
# ✅ Step 3: Use your local dog-cat-full-dataset-master.zip
local_zip = "dog-cat-full-dataset-master.zip"  # Already in your folder
extract_path = "dog-cat-full-dataset"

with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

DATA_DIR = os.path.join(extract_path, "dog-cat-full-dataset-master")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")


# ✅ Step 4: Load dataset
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# Prefetch to improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# ✅ Step 5: Build the CNN model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(160, 160, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ✅ Step 6: Train the model
EPOCHS = 3
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)

# ✅ Step 7: Visualization - Show some predictions
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in validation_dataset.take(1):
    predictions = model.predict(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        pred_label = "Dog 🐶" if predictions[i] > 0.5 else "Cat 🐱"
        true_label = class_names[labels[i]]
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
        plt.axis("off")
