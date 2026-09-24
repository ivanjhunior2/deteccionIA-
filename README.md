# Detector de rostros sintéticos

Prototipo en **Streamlit** que clasifica imágenes faciales como **reales** o **sintéticas** (generadas por IA) y muestra un mapa **Grad-CAM** con las regiones de la imagen que más influyeron en la decisión.

Desarrollado como parte de la monografía del **Diplomado en Ciencia de Datos — UMSS, 2026**.

## Qué se realizó

1. **Modelo de clasificación.** Se entrenó una red **ResNet50 con transferencia de aprendizaje** para clasificación binaria (`0 = Real`, `1 = Sintético`). El modelo desplegado corresponde al **escenario C**, entrenado con todos los generadores conocidos: FLUX.1 dev, FLUX.1 pro y SDXL. Se guardó en [models/model_selected.keras](models/model_selected.keras).
2. **Inferencia** ([src/inference.py](src/inference.py)). La imagen se convierte a RGB, se redimensiona a 256 × 256 y se aplica el `preprocess_input` de ResNet50, igual que en el entrenamiento. El modelo devuelve la probabilidad de que la imagen sea sintética, y se usa un umbral de decisión de **0.5**.
3. **Explicabilidad con Grad-CAM** ([src/gradcam.py](src/gradcam.py)). Se localiza el backbone ResNet50 dentro del modelo y se calculan los gradientes de la clase predicha respecto a la última capa convolucional (`conv5_block3_out`). El mapa resultante se superpone sobre la imagen original con el colormap `jet`.
4. **Aplicación web** ([app.py](app.py)). Permite subir una imagen (JPG/PNG) y muestra:
   - la clase predicha;
   - las probabilidades de que la imagen sea real o sintética;
   - la imagen original junto a su mapa Grad-CAM;
   - los detalles técnicos de la inferencia.
5. **Pruebas locales** ([test_local.py](test_local.py)). Clasifica todas las imágenes de [test_images/](test_images/) desde la consola. Esa carpeta contiene imágenes de prueba reales (`000xx.jpg`) y sintéticas de FLUX.1 dev, FLUX.1 pro y SDXL.

## Estructura

```
detector_streamlite/
├── app.py                  # Aplicación Streamlit
├── src/
│   ├── inference.py        # Carga del modelo, preprocesamiento y predicción
│   └── gradcam.py          # Cálculo y superposición del mapa Grad-CAM
├── models/
│   └── model_selected.keras  # Modelo ResNet50 (almacenado con Git LFS)
├── test_images/            # Imágenes de ejemplo reales y sintéticas
├── test_local.py           # Script de prueba por consola
└── requirements.txt
```

## Instalación y ejecución

El modelo (~200 MB) se almacena con **Git LFS**, así que Git LFS debe estar instalado antes de clonar el repositorio:

```bash
git lfs install
git clone https://github.com/ivanjhunior2/deteccionIA-.git
cd deteccionIA-
```

Crear un entorno virtual e instalar las dependencias (TensorFlow 2.20, Keras 3.13, Streamlit, etc.):

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

Probar el modelo por consola con las imágenes de ejemplo:

```bash
python test_local.py
```

## Limitaciones

Este sistema es un **prototipo académico**. Sus predicciones son una estimación del modelo, no una prueba definitiva de la autenticidad de una imagen. El mapa Grad-CAM muestra dónde se concentró la atención del modelo, pero no indica que haya identificado un artefacto o rasgo facial específico. El rendimiento puede bajar con generadores que no se usaron en el entrenamiento.
