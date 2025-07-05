import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

model_path = 'digit_model_cnn.h5'

# === MODEL SETUP ===
if os.path.exists(model_path):
    print("✅ Model already exists. Loading the saved CNN model...")
    model = load_model(model_path)
else:
    print(" No saved model found. Training CNN model from scratch...")

    # Load and preprocess dataset
    train_data = pd.read_csv('R:\ml/train.csv')
    X = train_data.iloc[:, 1:].values / 255.0  # Normalize
    X = X.reshape(-1, 28, 28, 1)
    y = to_categorical(train_data.iloc[:, 0], num_classes=10)

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    # Build CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        validation_data=(X_val, y_val),
                        epochs=20,
                        callbacks=[early_stop])

    model.save(model_path)
    print("Model trained and saved as", model_path)

    # Plot training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# === TESTING ON NEW IMAGE ===
print("\nTesting on a new handwritten digit image...")

import cv2

# Load and preprocess custom image
img = cv2.imread('R:\ml/num.png', cv2.IMREAD_GRAYSCALE)

# Resize to 28x28 while keeping aspect ratio
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)


# Normalize and reshape
img = img / 255.0
img = img.reshape(1, 28, 28, 1)

# Visualize what the model sees
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title("Preprocessed Input to Model")
plt.axis('off')
plt.show()

# Predict
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)
print(f" Predicted Digit: {predicted_digit}")
