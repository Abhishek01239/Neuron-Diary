import tensorflow as tf
import numpy as np

def get_valid_number():
    while True:
        try:
            value = float(input("Enter a number"))
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Load trained model
model = tf.keras.models.load_model("model.h5")

num = get_valid_number()

prediction = model.predict(np.array([[num]]))

print(f"🤖 AI Prediction: {prediction[0][0]}")
