import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv("data.csv")

x = data['input'].values
y = data['output'].values

# Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1]),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)

# Train model
model.fit(x, y, epochs=200)

# Save model
model.save("model.h5")

print("✅ Model trained and saved")
