import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("model.h5")

num = float(input("Enter a number: "))

prediction = model.predict(np.array([[num]]))

print(f"🤖 AI Prediction: {prediction[0][0]}")
