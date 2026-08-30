import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
source = os.path.join(BASE_DIR, "models", "fruit.keras")
target = os.path.join(BASE_DIR, "models", "fruit.tflite")

print("Converting:", source, flush=True)
model = tf.keras.models.load_model(source, compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(target, "wb") as f:
    f.write(tflite_model)
print("Created:", target, os.path.getsize(target), "bytes", flush=True)
