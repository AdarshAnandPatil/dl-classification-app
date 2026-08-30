from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    Interpreter = None


app = Flask(__name__)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fruit.tflite")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# --------------------------------------------------
# MODEL SETTINGS
# --------------------------------------------------

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "gif"
}

CLASS_NAMES = [
    "Apple",
    "Banana",
    "Grape",
    "Mango",
    "Orange"
]


# --------------------------------------------------
# TFLITE MODEL
# --------------------------------------------------

interpreter = None
model_error = None


def get_interpreter():

    global interpreter, model_error

    # Already loaded
    if interpreter is not None:
        return interpreter

    # Previous loading error
    if model_error is not None:
        raise RuntimeError(model_error)

    try:

        # Check TFLite runtime
        if Interpreter is None:
            raise RuntimeError(
                "tflite-runtime is not installed."
            )

        # Check model file
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                "TFLite model not found: " + MODEL_PATH
            )

        print("Loading TFLite model:")
        print(MODEL_PATH)

        # Load model
        interpreter = Interpreter(
            model_path=MODEL_PATH
        )

        # Allocate tensors
        interpreter.allocate_tensors()

        input_details = (
            interpreter.get_input_details()
        )

        output_details = (
            interpreter.get_output_details()
        )

        print("TFLite model loaded successfully!")

        print(
            "Input details:",
            input_details
        )

        print(
            "Output details:",
            output_details
        )

        # Check output classes
        output_shape = output_details[0]["shape"]

        output_count = int(
            output_shape[-1]
        )

        if output_count != len(CLASS_NAMES):

            raise RuntimeError(
                f"Model has {output_count} output classes, "
                f"but CLASS_NAMES has {len(CLASS_NAMES)} names."
            )

        return interpreter

    except Exception as e:

        model_error = (
            f"{type(e).__name__}: {e}"
        )

        print(
            "MODEL LOAD ERROR:",
            model_error
        )

        raise RuntimeError(
            model_error
        )


# --------------------------------------------------
# FILE CHECK
# --------------------------------------------------

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# --------------------------------------------------
# CLASSIFIER
# --------------------------------------------------

@app.route("/classifier")
def classifier():

    classifier_type = request.args.get(
        "type",
        "fruit"
    ).lower()

    return render_template(
        "classifier.html",
        classifier_type=classifier_type
    )


# --------------------------------------------------
# TEST SERVER
# --------------------------------------------------

@app.route("/test")
def test():

    return jsonify({

        "status": "working",

        "message":
            "Flask server is working correctly",

        "model_file_exists":
            os.path.isfile(MODEL_PATH),

        "model_loaded":
            interpreter is not None,

        "model_error":
            model_error

    })


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "ok"
    })


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    filepath = None

    try:

        # ------------------------------------------
        # LOAD MODEL
        # ------------------------------------------

        current_interpreter = (
            get_interpreter()
        )

        # ------------------------------------------
        # CHECK FILE
        # ------------------------------------------

        if "file" not in request.files:

            return jsonify(
                success=False,
                error=
                    "No image file received."
            ), 400

        file = request.files["file"]

        if (
            not file
            or file.filename == ""
        ):

            return jsonify(
                success=False,
                error=
                    "No image selected."
            ), 400

        if not allowed_file(
            file.filename
        ):

            return jsonify(
                success=False,
                error=
                    "Invalid image format. "
                    "Use PNG, JPG, JPEG or GIF."
            ), 400

        # ------------------------------------------
        # SECURE FILENAME
        # ------------------------------------------

        filename = secure_filename(
            file.filename
        )

        if not filename:

            return jsonify(
                success=False,
                error=
                    "Invalid filename."
            ), 400

        # ------------------------------------------
        # SAVE IMAGE
        # ------------------------------------------

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        print(
            "Image received:",
            filename
        )

        # ------------------------------------------
        # OPEN IMAGE
        # ------------------------------------------

        image = Image.open(
            filepath
        ).convert("RGB")

        # ------------------------------------------
        # GET MODEL INPUT/OUTPUT
        # ------------------------------------------

        input_details = (
            current_interpreter
            .get_input_details()
        )

        output_details = (
            current_interpreter
            .get_output_details()
        )

        input_shape = (
            input_details[0]["shape"]
        )

        # Expected:
        # [1, height, width, channels]

        if len(input_shape) != 4:

            raise RuntimeError(
                f"Unsupported model input shape: "
                f"{input_shape}"
            )

        image_height = int(
            input_shape[1]
        )

        image_width = int(
            input_shape[2]
        )

        # ------------------------------------------
        # RESIZE IMAGE
        # ------------------------------------------

        image = image.resize(
            (
                image_width,
                image_height
            )
        )

        # ------------------------------------------
        # CONVERT IMAGE TO NUMPY
        # ------------------------------------------

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ------------------------------------------
        # HANDLE MODEL INPUT TYPE
        # ------------------------------------------

        input_dtype = (
            input_details[0]["dtype"]
        )

        print(
            "Model input dtype:",
            input_dtype
        )

        if input_dtype == np.uint8:

            # Model expects 0-255
            image_array = (
                image_array
                .astype(np.uint8)
            )

        elif input_dtype == np.float32:

            # Model expects normalized 0-1
            image_array = (
                image_array / 255.0
            )

            image_array = (
                image_array.astype(
                    np.float32
                )
            )

        else:

            raise RuntimeError(
                f"Unsupported input dtype: "
                f"{input_dtype}"
            )

        # ------------------------------------------
        # SET INPUT TENSOR
        # ------------------------------------------

        current_interpreter.set_tensor(
            input_details[0]["index"],
            image_array
        )

        # ------------------------------------------
        # RUN MODEL
        # ------------------------------------------

        current_interpreter.invoke()

        # ------------------------------------------
        # GET PREDICTION
        # ------------------------------------------

        predictions = (
            current_interpreter
            .get_tensor(
                output_details[0]["index"]
            )[0]
        )

        # ------------------------------------------
        # FIND PREDICTED CLASS
        # ------------------------------------------

        predicted_index = int(
            np.argmax(predictions)
        )

        if (
            predicted_index
            >= len(CLASS_NAMES)
        ):

            raise RuntimeError(
                "Prediction index does not "
                "match CLASS_NAMES."
            )

        predicted_class = (
            CLASS_NAMES[
                predicted_index
            ]
        )

        # ------------------------------------------
        # CONFIDENCE
        # ------------------------------------------

        confidence = float(
            predictions[
                predicted_index
            ] * 100
        )

        # ------------------------------------------
        # ALL PROBABILITIES
        # ------------------------------------------

        probabilities = {

            class_name: round(
                float(
                    predictions[i]
                    * 100
                ),
                2
            )

            for i, class_name
            in enumerate(
                CLASS_NAMES
            )

        }

        # ------------------------------------------
        # RESPONSE
        # ------------------------------------------

        return jsonify(

            success=True,

            prediction=
                predicted_class,

            confidence=
                round(
                    confidence,
                    2
                ),

            probabilities=
                probabilities,

            filename=
                filename

        )

    # ----------------------------------------------
    # ERROR
    # ----------------------------------------------

    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return jsonify(

            success=False,

            error=
                f"{type(e).__name__}: {e}"

        ), 500

    # ----------------------------------------------
    # DELETE UPLOADED IMAGE
    # ----------------------------------------------

    finally:

        if (
            filepath
            and os.path.isfile(filepath)
        ):

            try:

                os.remove(
                    filepath
                )

            except OSError:

                pass


# --------------------------------------------------
# START SERVER
# --------------------------------------------------

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
