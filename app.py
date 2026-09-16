import streamlit as st
from PIL import Image

from src.inference import (
    load_model,
    predict_image,
    prepare_image,
)

from src.gradcam import (
    make_gradcam_heatmap,
    overlay_gradcam,
)


# ============================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================

st.set_page_config(
    page_title="Detector de rostros sintéticos",
    page_icon="🔍",
    layout="centered",
)


# ============================================================
# CARGA DEL MODELO
# ============================================================

@st.cache_resource
def get_model():
    """
    Carga el modelo una sola vez y lo mantiene en memoria.
    """
    return load_model()


try:
    model = get_model()
except Exception as error:
    st.error("No se pudo cargar el modelo.")
    st.exception(error)
    st.stop()


# ============================================================
# ENCABEZADO
# ============================================================

st.title("Detector de rostros sintéticos")

st.write(
    """
    Prototipo desarrollado para la clasificación de imágenes faciales
    como **reales** o **sintéticas**, utilizando un modelo
    **ResNet50 con transferencia de aprendizaje**.
    """
)

st.caption(
    "Modelo desplegado: ResNet50 — escenario C (todos los generadores conocidos)"
)

st.divider()


# ============================================================
# CARGA DE IMAGEN
# ============================================================

uploaded_file = st.file_uploader(
    "Selecciona una imagen facial",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception:
        st.error("No se pudo leer la imagen seleccionada.")
        st.stop()

    # --------------------------------------------------------
    # Mostrar imagen
    # --------------------------------------------------------

    st.subheader("Imagen seleccionada")

    st.image(
        image,
        caption="Imagen cargada",
        use_container_width=True,
    )

    st.write(
        f"Resolución original: **{image.width} × {image.height} px**"
    )

    st.divider()

    # ========================================================
    # BOTÓN DE ANÁLISIS
    # ========================================================

    if st.button(
        "Analizar imagen",
        type="primary",
        use_container_width=True,
    ):

        try:

            # ====================================================
            # INFERENCIA
            # ====================================================

            with st.spinner("Analizando imagen..."):

                result = predict_image(
                    model,
                    image,
                )

            fake_probability = result["fake_probability"]
            real_probability = result["real_probability"]
            predicted_class = result["class"]
            threshold = result["threshold"]

            # ====================================================
            # RESULTADO
            # ====================================================

            st.subheader("Resultado de la clasificación")

            if predicted_class == "Sintético":

                st.error(
                    "### Rostro clasificado como SINTÉTICO"
                )

            else:

                st.success(
                    "### Rostro clasificado como REAL"
                )

            # ----------------------------------------------------
            # Probabilidades
            # ----------------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    label="Probabilidad de ser sintético",
                    value=f"{fake_probability * 100:.2f} %",
                )

            with col2:

                st.metric(
                    label="Probabilidad de ser real",
                    value=f"{real_probability * 100:.2f} %",
                )

            st.write(
                f"Umbral de decisión utilizado: **{threshold:.2f}**"
            )

            # Barra de probabilidad de sintético
            st.write("Probabilidad estimada de contenido sintético:")

            st.progress(
                min(
                    max(float(fake_probability), 0.0),
                    1.0,
                )
            )

            st.divider()

            # ====================================================
            # GRAD-CAM
            # ====================================================

            st.subheader("Explicación visual con Grad-CAM")

            with st.spinner(
                "Generando mapa de activación Grad-CAM..."
            ):

                processed_image = prepare_image(
                    image
                )

                (
                    heatmap,
                    explained_class,
                    gradcam_probability,
                ) = make_gradcam_heatmap(
                    model,
                    processed_image,
                    threshold=threshold,
                )

                gradcam_image = overlay_gradcam(
                    image,
                    heatmap,
                    alpha=0.40,
                )

            # ----------------------------------------------------
            # Mostrar Grad-CAM
            # ----------------------------------------------------

            col_original, col_gradcam = st.columns(2)

            with col_original:

                st.image(
                    image,
                    caption="Imagen original",
                    use_container_width=True,
                )

            with col_gradcam:

                st.image(
                    gradcam_image,
                    caption=(
                        f"Grad-CAM — clase explicada: "
                        f"{explained_class}"
                    ),
                    use_container_width=True,
                )

            st.info(
                """
                **Interpretación de Grad-CAM:** las regiones con mayor
                intensidad representan zonas de la imagen que tuvieron
                mayor influencia en la decisión del modelo.

                Esta visualización permite observar la distribución
                espacial de la atención del modelo, pero no implica que
                este haya identificado explícitamente una característica
                facial o artefacto específico.
                """
            )

            st.divider()

            # ====================================================
            # INFORMACIÓN TÉCNICA
            # ====================================================

            with st.expander(
                "Ver información técnica de la inferencia"
            ):

                st.write(
                    "**Arquitectura:** ResNet50"
                )

                st.write(
                    "**Tamaño de entrada del modelo:** 256 × 256 × 3"
                )

                st.write(
                    "**Tipo de problema:** Clasificación binaria"
                )

                st.write(
                    "**Clase 0:** Real"
                )

                st.write(
                    "**Clase 1:** Sintético"
                )

                st.write(
                    f"**Umbral de decisión:** {threshold}"
                )

                st.write(
                    f"**Probabilidad sintética sin redondear:** "
                    f"{fake_probability:.10f}"
                )

                st.write(
                    f"**Probabilidad real sin redondear:** "
                    f"{real_probability:.10f}"
                )

                st.write(
                    f"**Clase explicada por Grad-CAM:** "
                    f"{explained_class}"
                )

        except Exception as error:

            st.error(
                "Se produjo un error durante el análisis de la imagen."
            )

            st.exception(error)


# ============================================================
# INFORMACIÓN FINAL
# ============================================================

st.divider()

st.warning(
    """
    **Uso académico:** este sistema corresponde a un prototipo
    experimental desarrollado en el contexto de una monografía.
    La predicción debe interpretarse como una estimación del modelo
    y no como una prueba definitiva de autenticidad de una imagen.
    """
)

st.caption(
    "Diplomado en Ciencia de Datos — UMSS, 2026"
)