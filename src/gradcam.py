import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib


THRESHOLD = 0.5
LAST_CONV_LAYER_NAME = "conv5_block3_out"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _call_layer(layer, x):
    """
    Ejecuta una capa en modo inferencia.
    """

    if isinstance(layer, tf.keras.layers.InputLayer):
        return x

    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def _contains_layer(model, layer_name):
    """
    Comprueba si un modelo contiene una capa determinada.
    """

    try:
        model.get_layer(layer_name)
        return True
    except ValueError:
        return False


def _find_backbone(model, layer_name=LAST_CONV_LAYER_NAME):
    """
    Busca específicamente el submodelo que contiene
    la última capa convolucional de ResNet50.

    Esto evita confundir ResNet50 con el modelo
    de data augmentation.
    """

    # --------------------------------------------------------
    # Primero buscamos entre los modelos directamente
    # contenidos en el modelo principal
    # --------------------------------------------------------

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            if _contains_layer(
                layer,
                layer_name
            ):
                return layer

    # --------------------------------------------------------
    # Búsqueda recursiva por si hubiera más niveles
    # --------------------------------------------------------

    def recursive_search(current_model):

        for layer in current_model.layers:

            if isinstance(layer, tf.keras.Model):

                if _contains_layer(
                    layer,
                    layer_name
                ):
                    return layer

                found = recursive_search(layer)

                if found is not None:
                    return found

        return None

    backbone = recursive_search(model)

    if backbone is None:

        raise ValueError(
            f"No se encontró ningún submodelo que contenga "
            f"la capa '{layer_name}'."
        )

    return backbone


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    model,
    image_array,
    threshold=THRESHOLD
):
    """
    Genera el mapa Grad-CAM para la clase predicha.

    Parámetros
    ----------
    model:
        Modelo completo cargado.

    image_array:
        Imagen preparada con shape:
        (1, 256, 256, 3)

    threshold:
        Umbral de decisión.

    Retorna
    -------
    heatmap
    explained_class
    fake_probability
    """

    # --------------------------------------------------------
    # Buscar ResNet50 correctamente
    # --------------------------------------------------------

    base_model = _find_backbone(
        model,
        LAST_CONV_LAYER_NAME
    )

    print(
        "Backbone encontrado:",
        base_model.name
    )

    # --------------------------------------------------------
    # Obtener última capa convolucional
    # --------------------------------------------------------

    last_conv_layer = base_model.get_layer(
        LAST_CONV_LAYER_NAME
    )

    # --------------------------------------------------------
    # Modelo auxiliar de Grad-CAM
    # --------------------------------------------------------

    conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ],
        name="gradcam_backbone"
    )

    # --------------------------------------------------------
    # Encontrar posición de ResNet dentro del modelo principal
    # --------------------------------------------------------

    try:

        base_index = model.layers.index(
            base_model
        )

    except ValueError:

        raise ValueError(
            "La ResNet50 fue encontrada, pero no está directamente "
            "contenida en el modelo principal. "
            "Se requiere revisar la estructura del modelo."
        )

    # --------------------------------------------------------
    # Preparar entrada
    # --------------------------------------------------------

    x = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Ejecutar las capas anteriores a ResNet50
    #
    # Aquí deberían estar, por ejemplo:
    # - augmentation
    # - preprocess_input
    # --------------------------------------------------------

    for layer in model.layers[:base_index]:

        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):
            continue

        x = _call_layer(
            layer,
            x
        )

    # ========================================================
    # PRIMERA PASADA:
    # Determinar qué clase predijo el modelo
    # ========================================================

    _, base_output = conv_model(
        x,
        training=False
    )

    prediction = base_output

    # Pasar por:
    # GlobalAveragePooling
    # Dropout
    # Dense sigmoid

    for layer in model.layers[
        base_index + 1:
    ]:

        prediction = _call_layer(
            layer,
            prediction
        )

    fake_probability = float(
        prediction.numpy().reshape(-1)[0]
    )

    explain_fake = (
        fake_probability >= threshold
    )

    # ========================================================
    # SEGUNDA PASADA:
    # Calcular gradientes
    # ========================================================

    with tf.GradientTape() as tape:

        conv_outputs, base_output = conv_model(
            x,
            training=False
        )

        tape.watch(
            conv_outputs
        )

        prediction = base_output

        for layer in model.layers[
            base_index + 1:
        ]:

            prediction = _call_layer(
                layer,
                prediction
            )

        fake_score = prediction[:, 0]

        if explain_fake:

            score = fake_score
            explained_class = "Sintético"

        else:

            score = 1.0 - fake_score
            explained_class = "Real"

    # --------------------------------------------------------
    # Gradientes
    # --------------------------------------------------------

    grads = tape.gradient(
        score,
        conv_outputs
    )

    if grads is None:

        raise RuntimeError(
            "TensorFlow no pudo calcular los gradientes "
            "necesarios para Grad-CAM."
        )

    # --------------------------------------------------------
    # Importancia promedio por canal
    # --------------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Primera imagen del batch
    conv_outputs = conv_outputs[0]

    # --------------------------------------------------------
    # Combinación ponderada
    # --------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs
        * pooled_grads,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    heatmap = tf.maximum(
        heatmap,
        0
    )

    # --------------------------------------------------------
    # Normalizar entre 0 y 1
    # --------------------------------------------------------

    maximum = tf.reduce_max(
        heatmap
    )

    if float(maximum) > 0:

        heatmap = (
            heatmap / maximum
        )

    return (
        heatmap.numpy(),
        explained_class,
        fake_probability
    )


# ============================================================
# SUPERPOSICIÓN DEL MAPA
# ============================================================

def overlay_gradcam(
    original_image,
    heatmap,
    alpha=0.40
):
    """
    Superpone el mapa Grad-CAM sobre la imagen original.
    """

    image = original_image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Convertir heatmap a imagen
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        image.size,
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # Aplicar mapa de colores
    # --------------------------------------------------------

    heatmap_array = (
        np.asarray(
            heatmap_image
        ).astype(np.float32)
        / 255.0
    )

    cmap = matplotlib.colormaps[
        "jet"
    ]

    colored_heatmap = cmap(
        heatmap_array
    )[:, :, :3]

    colored_heatmap = Image.fromarray(
        np.uint8(
            colored_heatmap
            * 255
        )
    )

    # --------------------------------------------------------
    # Superposición
    # --------------------------------------------------------

    overlay = Image.blend(
        image,
        colored_heatmap,
        alpha
    )

    return overlay