import os
import numpy as np
import pydicom
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 256
CASES_DIR = "cases"
EPOCHS = 15

def read_ct_images(ct_folder):
    imgs = []
    for fname in sorted(os.listdir(ct_folder)):
        if fname.endswith(".dcm"):
            ct = pydicom.dcmread(os.path.join(ct_folder, fname))
            img = cv2.resize(ct.pixel_array, (IMG_SIZE, IMG_SIZE))
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
            imgs.append(img)
    return np.array(imgs)

def read_mask(rtstruct, ct_images):
    # This is a placeholder; actual RTSTRUCT-to-mask requires parsing contours and rasterizing to image space
    # Libraries: dicompyler-core, plastimatch, or custom code (complex)
    # Here: generate dummy masks for code completeness
    return np.random.randint(0, 2, size=ct_images.shape, dtype="uint8")  # Replace with real RTSTRUCT parsing

X, y = [], []

for patient in os.listdir(CASES_DIR):
    ct_folder = os.path.join(CASES_DIR, patient, "CT")
    rtstruct_path = os.path.join(CASES_DIR, patient, "RTSTRUCT", "RTSTRUCT.dcm")
    ct_images = read_ct_images(ct_folder)
    mask = read_mask(rtstruct_path, ct_images)
    X.append(ct_images)
    y.append(mask)

X = np.concatenate(X)
y = np.concatenate(y)
X = X[..., np.newaxis]
y = y[..., np.newaxis]

# Build simple segmentation model (U-Net style)
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)
x = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, activation="sigmoid", padding="same")(x)
model = tf.keras.Model(inputs, x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, batch_size=8, epochs=EPOCHS, validation_split=0.1)
model.save("ct_tumor_seg_model.h5")
print("Model saved as ct_tumor_seg_model.h5")
