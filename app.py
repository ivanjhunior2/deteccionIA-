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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Estilos para que toda la interfaz quepa en una pantalla:
# - menos margen superior e inferior
# - imágenes limitadas a una fracción de la altura de la ventana
# - menos espacio entre elementos
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0.5rem;
        max-width: 100%;
    }
    h1 {
        padding-top: 0;
        padding-bottom: 0;
    }
    [data-testid="stImage"] img {
        max-height: 52vh;
        object-fit: contain;
    }
    [data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }
    [data-testid="stFileUploaderDropzone"] {
        padding: 0.5rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
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

col_title, col_info = st.columns([2, 3], vertical_alignment="center")

with col_title:
    st.title("Detector de rostros sintéticos")

with col_info:
    st.caption(
        "Clasificación de imágenes faciales como **reales** o "
        "**sintéticas** con un modelo **ResNet50 con transferencia de "
        "aprendizaje** — escenario C (todos los generadores conocidos). "
        "Diplomado en Ciencia de Datos — UMSS, 2026."
    )


# ============================================================
# DISTRIBUCIÓN EN TRES COLUMNAS
# ============================================================

col_input, col_result, col_gradcam = st.columns(
    [1, 1, 1],
    gap="medium",
    border=True,
)


# ============================================================
# COLUMNA 1: CARGA DE IMAGEN
# ============================================================

with col_input:

    st.subheader("1. Imagen")

    uploaded_file = st.file_uploader(
        "Selecciona una imagen facial",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

        except Exception:
            st.error("No se pudo leer la imagen seleccionada.")

    if image is not None:

        st.image(
            image,
            caption=(
                f"Imagen cargada — "
                f"{image.width} × {image.height} px"
            ),
            width="stretch",
        )

    analyze = st.button(
        "Analizar imagen",
        type="primary",
        width="stretch",
        disabled=image is None,
    )


# ============================================================
# ANÁLISIS
# ============================================================

with col_result:
    st.subheader("2. Resultado")

with col_gradcam:
    st.subheader("3. Explicación Grad-CAM")


if image is None or not analyze:

    with col_result:
        st.info(
            "Carga una imagen y presiona **Analizar imagen** "
            "para ver la clasificación."
        )

    with col_gradcam:
        st.info(
            "Aquí se mostrará el mapa de activación Grad-CAM "
            "de la imagen analizada."
        )

else:

    try:

        # ====================================================
        # INFERENCIA
        # ====================================================

        with col_result:

            with st.spinner("Analizando imagen..."):

                result = predict_image(
                    model,
                    image,
                )

            fake_probability = result["fake_probability"]
            real_probability = result["real_probability"]
            predicted_class = result["class"]
            threshold = result["threshold"]

            # ------------------------------------------------
            # Resultado
            # ------------------------------------------------

            if predicted_class == "Sintético":

                st.error(
                    "#### Rostro clasificado como SINTÉTICO"
                )

            else:

                st.success(
                    "#### Rostro clasificado como REAL"
                )

            # ------------------------------------------------
            # Probabilidades
            # ------------------------------------------------

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

            # Barra de probabilidad de sintético
            st.progress(
                min(
                    max(float(fake_probability), 0.0),
                    1.0,
                ),
                text=(
                    f"Probabilidad de contenido sintético "
                    f"(umbral de decisión: {threshold:.2f})"
                ),
            )

        # ====================================================
        # GRAD-CAM
        # ====================================================

        with col_gradcam:

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

            st.image(
                gradcam_image,
                caption=(
                    f"Grad-CAM — clase explicada: "
                    f"{explained_class}"
                ),
                width="stretch",
            )

            with st.expander("¿Cómo interpretar Grad-CAM?"):

                st.write(
                    """
                    Las regiones con mayor intensidad representan
                    zonas de la imagen que tuvieron mayor influencia
                    en la decisión del modelo.

                    Esta visualización permite observar la
                    distribución espacial de la atención del modelo,
                    pero no implica que este haya identificado
                    explícitamente una característica facial o
                    artefacto específico.
                    """
                )

        # ====================================================
        # INFORMACIÓN TÉCNICA
        # ====================================================

        with col_result:

            with st.expander(
                "Ver información técnica de la inferencia"
            ):

                st.write(
                    f"""
                    - **Arquitectura:** ResNet50
                    - **Tamaño de entrada del modelo:** 256 × 256 × 3
                    - **Tipo de problema:** Clasificación binaria
                    - **Clase 0:** Real — **Clase 1:** Sintético
                    - **Umbral de decisión:** {threshold}
                    - **Probabilidad sintética sin redondear:** {fake_probability:.10f}
                    - **Probabilidad real sin redondear:** {real_probability:.10f}
                    - **Clase explicada por Grad-CAM:** {explained_class}
                    """
                )

    except Exception as error:

        with col_result:

            st.error(
                "Se produjo un error durante el análisis de la imagen."
            )

            st.exception(error)


# ============================================================
# INFORMACIÓN FINAL
# ============================================================

st.caption(
    "⚠️ **Uso académico:** este sistema corresponde a un prototipo "
    "experimental desarrollado en el contexto de una monografía. "
    "La predicción debe interpretarse como una estimación del modelo "
    "y no como una prueba definitiva de autenticidad de una imagen."
)
