from pathlib import Path
import random
import re
import shutil

original = Path("/home/julia/Escritorio/projects/brats_synthesis/data/BraTS/processed/brats_custom_synthesis_t1n_seg_to_t1c/resvit_t1_seg_to_t1ce")
reduced = Path("/home/julia/Escritorio/projects/brats_synthesis/data/BraTS/processed/brats_custom_synthesis_t1n_seg_to_t1c/resvit_t1_seg_to_t1ce/t1n_seg_10pct")

seed = 2026
fraction = 0.10

# Accepta PNG, JPG i NPY
extensions = {".png", ".jpg", ".jpeg", ".npy"}

def patient_id(path):
    # Exemple: BraTS-GLI-00045-000_z043.png
    return re.sub(r"_z\d+$", "", path.stem)

train_files = [
    p for p in (original / "train").rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

patients = sorted({patient_id(p) for p in train_files})

rng = random.Random(seed)
rng.shuffle(patients)

n_patients = max(1, round(len(patients) * fraction))
selected_patients = set(patients[:n_patients])

print(f"Pacients train totals: {len(patients)}")
print(f"Pacients seleccionats: {len(selected_patients)}")
print(f"Slices train totals: {len(train_files)}")

# Copiar train: només el 10% de pacients
for src in train_files:
    if patient_id(src) in selected_patients:
        dst = reduced / "train" / src.relative_to(original / "train")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

# Copiar val i test sencers
for split in ("val", "test"):
    src_dir = original / split
    if src_dir.exists():
        shutil.copytree(src_dir, reduced / split, dirs_exist_ok=True)

print(f"Dataset creat a: {reduced}")
