
from pathlib import Path
import csv
import random
from collections import Counter

DATASET_DIR = Path("data/raw/trashnet")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

rng = random.Random(SEED)
manifest = []

for class_name in CLASSES:
    class_dir = DATASET_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Missing class folder: {class_dir}"
        )

    images = sorted(
        path for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    )

    if len(images) < 3:
        raise ValueError(
            f"Not enough images in class: {class_name}"
        )

    rng.shuffle(images)

    total = len(images)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    for index, image_path in enumerate(images):
        if index < train_end:
            split = "train"
        elif index < val_end:
            split = "validation"
        else:
            split = "test"

        manifest.append({
            "file": str(image_path),
            "class": class_name,
            "split": split,
        })

output_path = REPORT_DIR / "dataset_split.csv"

with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["file", "class", "split"],
    )
    writer.writeheader()
    writer.writerows(manifest)

print("Dataset split manifest created.")
print(f"Random seed: {SEED}")
print(f"Total images assigned: {len(manifest)}")
print(f"Manifest: {output_path}\n")

counts = Counter(
    (row["class"], row["split"]) for row in manifest
)

for class_name in CLASSES:
    print(
        f"{class_name:10} "
        f"train={counts[(class_name, 'train')]:3} "
        f"validation={counts[(class_name, 'validation')]:3} "
        f"test={counts[(class_name, 'test')]:3}"
    )

print("\nSplit totals:")
for split in ["train", "validation", "test"]:
    print(
        f"{split}: "
        f"{sum(row['split'] == split for row in manifest)}"
    )
