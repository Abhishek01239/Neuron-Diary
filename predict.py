import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import logging
import csv
import config
from datetime import datetime

# ---------------- LOGGING SETUP ----------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("🚀 Prediction script started")

# ---------------- FUNCTIONS ----------------
def load_latest_model():
    model_files = sorted(
        [f for f in os.listdir(config.MODEL_DIR) if f.endswith(".h5")]
    )
    if not model_files:
        raise FileNotFoundError("No trained model found.")

    model_path = os.path.join(config.MODEL_DIR, model_files[-1])
    model = tf.keras.models.load_model(model_path, compile=False)

    logging.info(f"Loaded model: {model_files[-1]}")
    return model, model_files[-1]

def get_user_numbers():
    user_input = input("🔢 Enter numbers separated by commas: ").strip()

    if not user_input:
        raise ValueError("Input cannot be empty.")

    try:
        nums = [float(x.strip()) for x in user_input.split(",")]
        logging.info(f"User input: {nums}")
        return np.array(nums).reshape(-1, 1)
    except ValueError:
        raise ValueError("Please enter valid numeric values only.")

def validate_range(nums, x_min, x_max):
    out = [n for n in nums.flatten() if n < x_min or n > x_max]
    if out:
        raise ValueError(
            f"Inputs {out} are outside training range [{x_min}, {x_max}]"
        )

def save_history(inputs, outputs, model_name):
    file_exists = os.path.isfile(config.HISTORY_FILE)

    with open(config.HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "model", "input", "prediction"])

        for i, o in zip(inputs.flatten(), outputs.flatten()):
            writer.writerow([
                datetime.now().isoformat(),
                model_name,
                i,
                round(o, config.MAX_DECIMALS)
            ])

    logging.info("Prediction history saved")

# ---------------- MAIN FLOW ----------------
try:
    model, model_name = load_latest_model()
    x_min, x_max, y_min, y_max = np.load(config.SCALER_PATH)

    nums = get_user_numbers()
    validate_range(nums, x_min, x_max)

    nums_norm = (nums - x_min) / (x_max - x_min)
    pred_norm = model.predict(nums_norm)
    preds = pred_norm * (y_max - y_min) + y_min

    print(f"\n🤖 AI Predictions (using {model_name}):")
    for i, p in zip(nums.flatten(), preds.flatten()):
        print(f"{i} → {round(p, config.MAX_DECIMALS)}")

    save_history(nums, preds, model_name)

    if config.PLOT_RESULTS:
        plt.figure()
        plt.scatter(nums, preds)
        plt.xlabel("Input")
        plt.ylabel("Prediction")
        plt.title("AI Prediction Output")
        plt.show()

    logging.info("Prediction completed successfully")

except Exception as e:
    logging.error(str(e))
    print(f"\n❌ Error: {e}")
    print(config.ERROR_TIPS)
    sys.exit(1)
