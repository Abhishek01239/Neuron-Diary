import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import config

def load_latest_model():
    model_files = sorted(
        [f for f in os.listdir(config.MODEL_DIR) if f.endswith(".h5")]
    )
    if not model_files:
        raise FileNotFoundError("No trained model found.")

    model_path = os.path.join(config.MODEL_DIR, model_files[-1])
    return tf.keras.models.load_model(model_path, compile=False), model_files[-1]

def get_user_numbers():
    user_input = input("🔢 Enter numbers separated by commas: ").strip()

    if not user_input:
        raise ValueError("Input cannot be empty.")

    try:
        nums = [float(x.strip()) for x in user_input.split(",")]
        return np.array(nums).reshape(-1, 1)
    except ValueError:
        raise ValueError("Please enter valid numeric values only.")

def validate_range(nums, x_min, x_max):
    out_of_range = [n for n in nums.flatten() if n < x_min or n > x_max]
    if out_of_range:
        raise ValueError(
            f"Inputs {out_of_range} are outside training range [{x_min}, {x_max}]"
        )

# ================= MAIN =================
try:
    model, model_name = load_latest_model()
    x_min, x_max, y_min, y_max = np.load(config.SCALER_PATH)

    nums = get_user_numbers()
    validate_range(nums, x_min, x_max)

    # Normalize input
    nums_norm = (nums - x_min) / (x_max - x_min)

    # Predict
    pred_norm = model.predict(nums_norm)
    preds = pred_norm * (y_max - y_min) + y_min

    print(f"\n🤖 AI Predictions (using {model_name}):")
    for i, p in zip(nums.flatten(), preds.flatten()):
        print(f"{i} → {round(p, config.MAX_DECIMALS)}")

    # Plot results
    if config.PLOT_RESULTS:
        plt.figure()
        plt.scatter(nums, preds)
        plt.xlabel("Input")
        plt.ylabel("Prediction")
        plt.title("AI Prediction Output")
        plt.show()

except Exception as e:
    print(f"\n❌ Error: {e}")
    print(config.ERROR_TIPS)
    sys.exit(1)
