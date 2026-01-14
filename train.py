import tensorflow as tf
import numpy as np

# Training data
x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 3, 4, 5, 6], dtype=float)

# Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
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
