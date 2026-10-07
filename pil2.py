from PIL import Image
from pathlib import Path

carpeta = Path(r"/home/julia/Escritorio/projects/brats_synthesis/data/BraTS/processed/brats_custom_synthesis_t1n_seg_to_t1c/t1n_seg_10pct/train")

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

fitxers = [
    f for f in carpeta.rglob("*")
    if f.is_file() and f.suffix.lower() in extensions
]

print(f"Imatges trobades: {len(fitxers)}")

for fitxer in fitxers:
    try:
        with Image.open(fitxer) as imatge:
            imatge.load()
    except Exception as error:
        print(f"ERROR: {fitxer}")
        print(error)
