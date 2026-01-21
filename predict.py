import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def get_latest_model():
    model_files = sorted(
        [f for f in os.listdit("model") if f.endswith(".h5")]
    )
    return os.path.join("model", model_files[-1])

def get_numbers():
    while True:
        user_input = input(
            "Enter numbers separated by commas (e.g. 1, 5, 10):"
        )
        try:
            numbers = [float(x.strip()) for x in user_input.split(",")]
            return np.array(numbers).reshape(-1,1)
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Load trained model
model = tf.keras.models.load_model("model.h5")

inputs = get_numbers()
predictions = model.predict(inputs)

print("\n🤖 AI Prediction: ")
for i, pred in zip(inputs.flatten(), predictions.flatten()):
    print(f"{i}->{pred:.2f}")

plt.figure()
plt.scatter(inputs, predictions)
plt.xlabel("Input Numbers")
plt.ylabel("Prediction Output")
plt.title("AI Number Prediction Visualization")
plt.show()