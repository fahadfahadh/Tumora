import os
import numpy as np
import pydicom
import cv2
import tensorflow as tf

IMG_SIZE = 256
CASES_DIR = "cases"
MODEL_PATH = "ct_tumor_seg_model.h5"

def read_ct_images(ct_folder):
    imgs = []
    for fname in sorted(os.listdir(ct_folder)):
        if fname.endswith(".dcm"):
            ct = pydicom.dcmread(os.path.join(ct_folder, fname))
            img = cv2.resize(ct.pixel_array, (IMG_SIZE, IMG_SIZE))
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)
            imgs.append(img)
    return np.array(imgs)

model = tf.keras.models.load_model(MODEL_PATH)

for patient in os.listdir(CASES_DIR):
    ct_folder = os.path.join(CASES_DIR, patient, "CT")
    ct_images = read_ct_images(ct_folder)
    ct_images = ct_images[..., np.newaxis]
    preds = model.predict(ct_images)
    print(f"Patient {patient}: Prediction shape {preds.shape}")
    # Logic to save or display segmentation results goes here


import matplotlib.pyplot as plt
# Example: Show first slice and its mask
plt.subplot(1,2,1)
plt.imshow(ct_images[0, ..., 0], cmap='gray')
plt.title("CT Slice")
plt.subplot(1,2,2)
plt.imshow(preds[0, ..., 0] > 0.5, cmap='Reds', alpha=0.5)
plt.title("Predicted Mask")
plt.show()