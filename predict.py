import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

def load_letest_model():
    try:
        model_files = sorted([f for f in os.listdir("model") if f.endswith(".h5")])

        if not model_files:
            raise FileNotFoundError("No trained model found")
        return tf.keras.models.load_model(
        os.path.join("model", model_files[-1]),
                                        compile = False)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
def get_user_numbers():
    user_input = input("Enter numbers separated by commas:").strip()
    if not user_input:
        raise ValueError("Inpur cannot be empty")
    
    try:
        nums = [float(x.strip()) for x in user_input.split(",")]
        return np.array(nums).reshape(-1,1)
    except ValueError:
        raise ValueError("Please enter valid numbers only.")

def validate_range(nums, x_min, x_max):
    out_of_range = [n for n in nums.flatten() if n<x_min or n>x_max]
    if out_of_range:
        raise ValueError(f"Inputs {out_of_range} are outside training range [{x_min}, {x_max}]")

try:
    model = load_letest_model()
    x_min, x_max, y_min, y_max = np.load("model/scaler.npy")

    nums = get_user_numbers()
    validate_range(nums,x_min, x_max)

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

except Exception as e:
    print(f"\n Error: {e}")
    print("Tip: Use numbers within the training data range")