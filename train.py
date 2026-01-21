import tensorflow as tf
import numpy as np
import pandas as pd
import os

data = pd.read_csv("data.csv")

x = data['input'].values
y = data['output'].values

MODEL_DIR = "model"
MODEL_VERSION = "model_v1.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_VERSION)

os.makedirs(MODEL_DIR, exist_ok=True)

#  Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16,activation = 'relu', input_shape=[1]),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# Train model
model.fit(x, y, epochs=200)

# Save model
model.save(MODEL_PATH)

print("✅ Model trained and saved")
