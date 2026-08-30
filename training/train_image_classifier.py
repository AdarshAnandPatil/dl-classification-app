import tensorflow as tf
from pathlib import Path

# ==============================
# Fruit Image Classifier Training
# ==============================

DATASET_DIR = "fruit_dataset"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "fruit.keras"


# ------------------------------
# Load training dataset
# ------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# ------------------------------
# Load validation dataset
# ------------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


# Get class names
class_names = train_dataset.class_names

print("\nFruit Classes:")

for i, name in enumerate(class_names):
    print(i, "=", name)


# ------------------------------
# Improve dataset performance
# ------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)


# ------------------------------
# Data Augmentation
# ------------------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


# ------------------------------
# CNN Model
# ------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(160, 160, 3)
    ),

    data_augmentation,

    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ------------------------------
# Compile
# ------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ------------------------------
# Train
# ------------------------------

print("\nStarting Fruit Classifier Training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ------------------------------
# Save model
# ------------------------------

model.save(MODEL_PATH)

print("\n==============================")
print("Training completed!")
print("Model saved:")
print(MODEL_PATH)
print("==============================")
