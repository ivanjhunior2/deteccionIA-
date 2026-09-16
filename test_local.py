from pathlib import Path
from PIL import Image

from src.inference import load_model, predict_image


model = load_model()

TEST_DIR = Path("test_images")

for file in TEST_DIR.iterdir():

    if file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    image = Image.open(file)

    result = predict_image(model, image)

    print("-" * 60)
    print("Archivo:", file.name)
    print("Predicción:", result["class"])
    print(
        "Probabilidad sintética:",
        f"{result['fake_probability'] * 100:.6f}%"
    )
    print(
        "Probabilidad real:",
        f"{result['real_probability'] * 100:.6f}%"
    )