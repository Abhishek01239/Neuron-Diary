import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import config

# Load dataset
data = pd.read_csv("data.csv")
x = data["input"].values.astype(float)
y = data["output"].values.astype(float)

# Normalize data (min-max)
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)

# Ensure model directory exists
os.makedirs(config.MODEL_DIR, exist_ok=True)

# Save scaler values
np.save(config.SCALER_PATH, np.array([x_min, x_max, y_min, y_max]))

# Model versioning
existing_models = [
    f for f in os.listdir(config.MODEL_DIR) if f.endswith(".h5")
]
version = len(existing_models) + 1
model_name = f"model_v{version}.h5"
model_path = os.path.join(config.MODEL_DIR, model_name)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=[1]),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.MeanSquaredError()
)

# Train model
history = model.fit(x_norm, y_norm, epochs=200, verbose=0)

# Save model
model.save(model_path)

# Plot training loss
plt.figure()
plt.plot(history.history["loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss (Normalized Data)")
plt.show()

print(f"✅ Model trained and saved as {model_name}")
print(f"📊 Training range: [{x_min}, {x_max}]")
