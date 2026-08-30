from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fruit.tflite")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

CLASS_NAMES = [
    "Apple",
    "Banana",
    "Grape",
    "Mango",
    "Orange"
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# TFLite interpreter
interpreter = None
model_error = None


def get_interpreter():
    global interpreter, model_error

    if interpreter is not None:
        return interpreter

    if model_error is not None:
        raise RuntimeError(model_error)

    try:
        from tflite_runtime.interpreter import Interpreter

        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                "TFLite model not found: " + MODEL_PATH
            )

        print("Loading TFLite model:", MODEL_PATH)

        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()

        print("TFLite model loaded successfully!")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("Input details:", input_details)
        print("Output details:", output_details)

        output_count = output_details[0]["shape"][-1]

        if int(output_count) != len(CLASS_NAMES):
            raise RuntimeError(
                f"Model has {output_count} output classes, "
                f"but CLASS_NAMES has {len(CLASS_NAMES)} names."
            )

        return interpreter

    except Exception as e:
        model_error = f"{type(e).__name__}: {e}"
        print("MODEL LOAD ERROR:", model_error)
        raise RuntimeError(model_error)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/classifier")
def classifier():
    classifier_type = request.args.get(
        "type", "fruit"
    ).lower()

    return render_template(
        "classifier.html",
        classifier_type=classifier_type
    )


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
    return jsonify({
        "status": "ok"
    })


@app.route("/predict", methods=["POST"])
def predict():

    filepath = None

    try:
        current_interpreter = get_interpreter()

        if "file" not in request.files:
            return jsonify(
                success=False,
                error="No image file received."
            ), 400

        file = request.files["file"]

        if not file or file.filename == "":
            return jsonify(
                success=False,
                error="No image selected."
            ), 400

        if not allowed_file(file.filename):
            return jsonify(
                success=False,
                error="Invalid image format. Use PNG, JPG, JPEG or GIF."
            ), 400

        filename = secure_filename(file.filename)

        if not filename:
            return jsonify(
                success=False,
                error="Invalid filename."
            ), 400

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        print("Image received:", filename)

        image = Image.open(filepath).convert("RGB")

        input_details = current_interpreter.get_input_details()
        output_details = current_interpreter.get_output_details()

        input_shape = input_details[0]["shape"]

        image_height = int(input_shape[1])
        image_width = int(input_shape[2])

        print(
            "Required image size:",
            image_width,
            "x",
            image_height
        )

        image = image.resize(
            (image_width, image_height)
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        # Most image classification models use 0-1 input.
        image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Set input tensor
        current_interpreter.set_tensor(
            input_details[0]["index"],
            image_array
        )

        # Run model
        current_interpreter.invoke()

        # Get prediction
        predictions = current_interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        predictions = np.asarray(
            predictions,
            dtype=np.float32
        )

        predicted_index = int(
            np.argmax(predictions)
        )

        if predicted_index >= len(CLASS_NAMES):
            raise RuntimeError(
                "Model prediction index does not match CLASS_NAMES."
            )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index] * 100
        )

        probabilities = {
            class_name: round(
                float(predictions[i] * 100),
                2
            )
            for i, class_name in enumerate(CLASS_NAMES)
        }

        return jsonify(
            success=True,
            prediction=predicted_class,
            confidence=round(
                confidence,
                2
            ),
            probabilities=probabilities,
            filename=filename
        )

    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return jsonify(
            success=False,
            error=f"{type(e).__name__}: {e}"
        ), 500

    finally:

        if filepath and os.path.isfile(filepath):

            try:
                os.remove(filepath)

            except OSError:
                pass


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
