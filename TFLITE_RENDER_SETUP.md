# Render setup for the lightweight TFLite version

1. Root Directory: leave blank.
2. Build Command:
   `pip install -r requirements.txt && pip install tensorflow-cpu==2.16.2 && python convert_model.py`
3. Start Command:
   `gunicorn app:app`
4. Deploy. The build converts `models/fruit.keras` to `models/fruit.tflite`.
5. Runtime uses `tflite-runtime`, not full TensorFlow.

If the build fails during conversion, send the Render build log. Do not change the Start Command.
