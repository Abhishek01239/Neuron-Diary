import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Load latest model
model_files = sorted([f for f in os.listdir("model") if f.endswith(".h5")])
model = tf.keras.models.load_model(os.path.join("model", model_files[-1]),
                                   compile = False)

# Load scaler
x_min, x_max, y_min, y_max = np.load("model/scaler.npy")

# Input
user_input = input("🔢 Enter numbers separated by commas: ")
nums = np.array([float(x.strip()) for x in user_input.split(",")]).reshape(-1, 1)

# Normalize input
nums_norm = (nums - x_min) / (x_max - x_min)

# Predict
pred_norm = model.predict(nums_norm)

# De-normalize output
pred = pred_norm * (y_max - y_min) + y_min

print("\n🤖 AI Predictions:")
for i, p in zip(nums.flatten(), pred.flatten()):
    print(f"{i} → {p:.2f}")

# Plot
plt.figure()
plt.scatter(nums, pred)
plt.xlabel("Input")
plt.ylabel("Prediction")
plt.title("Normalized Model Prediction")
plt.show()
