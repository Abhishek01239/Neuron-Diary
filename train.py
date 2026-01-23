import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data.csv")
x = data["input"].values.astype(float)
y = data["output"].values.astype(float)

# Normalize (min-max scaling)
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)

# Save normalization values
np.save("model/scaler.npy", np.array([x_min, x_max, y_min, y_max]))

# Model versioning
MODEL_DIR = "model"
MODEL_VERSION = "model_v3.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_VERSION)
os.makedirs(MODEL_DIR, exist_ok=True)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=[1]),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

# Train
history = model.fit(x_norm, y_norm, epochs=500, verbose=0)

# Save model
model.save(MODEL_PATH)

# Plot loss
plt.figure()
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Normalized Training Loss")
plt.show()

print(f"✅ Model trained with normalization and saved as {MODEL_VERSION}")
