from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model_selected.keras"
)


def load_model():
    """
    Carga el modelo ResNet50 seleccionado durante el proyecto.
    """
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


def prepare_image(image: Image.Image):
    """
    Aplica el mismo esquema de entrada empleado durante el modelado:
    - RGB
    - 256 x 256
    - float32
    - preprocess_input de ResNet50
    """

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    # Agregar dimensión batch:
    # (256, 256, 3) -> (1, 256, 256, 3)
    array = np.expand_dims(array, axis=0)

    # Preprocesamiento propio de ResNet50
    array = preprocess_input(array)

    return array


def predict_image(model, image: Image.Image):
    """
    Devuelve probabilidad de imagen sintética
    y clase predicha.
    """

    processed = prepare_image(image)

    prediction = model.predict(
        processed,
        verbose=0
    )

    fake_probability = float(
        prediction.reshape(-1)[0]
    )

    real_probability = 1.0 - fake_probability

    predicted_class = (
        "Sintético"
        if fake_probability >= THRESHOLD
        else "Real"
    )

    return {
        "class": predicted_class,
        "fake_probability": fake_probability,
        "real_probability": real_probability,
        "threshold": THRESHOLD,
    }