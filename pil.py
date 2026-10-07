from PIL import Image
from pathlib import Path

carpeta = Path("/home/julia/Escritorio/projects/brats_synthesis/data/BraTS/processed/brats_custom_synthesis_t1n_seg_to_t1c/t1n_seg_10pct/train")

for fitxer in carpeta.rglob("*"):
    if fitxer.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        try:
            with Image.open(fitxer) as imatge:
                imatge.verify()
        except Exception as error:
            print(f"Imatge amb error: {fitxer}")
            print(f"Error: {error}")
