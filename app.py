from flask import Flask, render_template, request, jsonify
import os
import gc
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fruit.tflite")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
CLASS_NAMES = ["Apple", "Banana", "Grape", "Mango", "Orange"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

interpreter = None
input_details = None
output_details = None
model_error = None


def get_interpreter():
    global interpreter, input_details, output_details, model_error
    if interpreter is not None:
        return interpreter
    if model_error is not None:
        raise RuntimeError(model_error)
    try:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError("TFLite model not found: " + MODEL_PATH)
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            # Allows local testing when TensorFlow is installed.
            from tensorflow.lite import Interpreter
        interpreter = Interpreter(model_path=MODEL_PATH, num_threads=1)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model loaded", flush=True)
        print("Input:", input_details[0], flush=True)
        print("Output:", output_details[0], flush=True)
        if output_details[0]["shape"][-1] != len(CLASS_NAMES):
            raise RuntimeError("TFLite model output classes do not match CLASS_NAMES")
        return interpreter
    except Exception as e:
        model_error = f"{type(e).__name__}: {e}"
        print("MODEL LOAD ERROR:", model_error, flush=True)
        raise RuntimeError(model_error)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_input(image, detail):
    shape = detail["shape"]
    if len(shape) != 4:
        raise RuntimeError(f"Unsupported model input shape: {shape}")
    height, width = int(shape[1]), int(shape[2])
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    data = np.asarray(image, dtype=np.float32) / 255.0
    data = np.expand_dims(data, axis=0)

    dtype = detail["dtype"]
    if dtype == np.float32:
        return data.astype(np.float32)

    scale, zero_point = detail.get("quantization", (0.0, 0))
    if scale and np.issubdtype(dtype, np.integer):
        data = np.round(data / scale + zero_point)
        info = np.iinfo(dtype)
        data = np.clip(data, info.min, info.max)
        return data.astype(dtype)

    return data.astype(dtype)


def read_output(detail, raw):
    values = np.asarray(raw, dtype=np.float32)
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if scale:
        values = (values - zero_point) * scale
    return values


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/classifier")
def classifier():
    classifier_type = request.args.get("type", "fruit").lower()
    return render_template("classifier.html", classifier_type=classifier_type)

@app.route("/test")
def test():
    return jsonify({
        "status": "working",
        "message": "Flask server is working correctly",
        "model_file_exists": os.path.isfile(MODEL_PATH),
        "model_loaded": interpreter is not None,
        "model_error": model_error
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    filepath = None
    try:
        if "file" not in request.files:
            return jsonify(success=False, error="No image file received."), 400
        file = request.files["file"]
        if not file or not file.filename:
            return jsonify(success=False, error="No image selected."), 400
        if not allowed_file(file.filename):
            return jsonify(success=False, error="Invalid image format. Use PNG, JPG, JPEG or GIF."), 400

        filename = secure_filename(file.filename)
        if not filename:
            return jsonify(success=False, error="Invalid filename."), 400
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        current = get_interpreter()
        detail = input_details[0]
        image = Image.open(filepath).convert("RGB")
        tensor = prepare_input(image, detail)
        current.set_tensor(detail["index"], tensor)
        current.invoke()
        values = read_output(output_details[0], current.get_tensor(output_details[0]["index"])[0])

        predicted_index = int(np.argmax(values))
        if predicted_index >= len(CLASS_NAMES):
            raise RuntimeError("Prediction index does not match CLASS_NAMES")

        # Softmax only when the model output is logits rather than probabilities.
        exp_values = np.exp(values - np.max(values))
        probabilities_array = exp_values / np.sum(exp_values)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities_array[predicted_index] * 100)
        probabilities = {name: round(float(probabilities_array[i] * 100), 2) for i, name in enumerate(CLASS_NAMES)}

        return jsonify(success=True, prediction=predicted_class,
                       confidence=round(confidence, 2),
                       probabilities=probabilities, filename=filename)
    except Exception as e:
        print("Prediction error:", repr(e), flush=True)
        return jsonify(success=False, error=f"{type(e).__name__}: {e}"), 500
    finally:
        if filepath and os.path.isfile(filepath):
            try: os.remove(filepath)
            except OSError: pass
        gc.collect()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
