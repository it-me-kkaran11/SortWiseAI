
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import csv

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

total_checked = 0
total_valid = 0
total_invalid = 0
report_rows = []

print("Checking dataset images...")
print("Original images will not be changed.\n")

for class_name in CLASSES:
    class_dir = DATASET_DIR / class_name

    if not class_dir.is_dir():
        print(f"ERROR: Missing class folder: {class_dir}")
        continue

    class_checked = 0
    class_valid = 0
    class_invalid = 0

    for image_path in class_dir.rglob("*"):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        class_checked += 1
        total_checked += 1

        try:
            with Image.open(image_path) as image:
                image.verify()

            class_valid += 1
            total_valid += 1

            report_rows.append({
                "file": str(image_path),
                "class": class_name,
                "status": "valid",
                "error": "",
            })

        except (UnidentifiedImageError, OSError, ValueError) as error:
            class_invalid += 1
            total_invalid += 1

            report_rows.append({
                "file": str(image_path),
                "class": class_name,
                "status": "invalid",
                "error": str(error),
            })

    print(
        f"{class_name}: checked={class_checked}, "
        f"valid={class_valid}, invalid={class_invalid}"
    )

report_path = REPORT_DIR / "dataset_validation.csv"

with report_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["file", "class", "status", "error"],
    )
    writer.writeheader()
    writer.writerows(report_rows)

print("\n--- DATASET VALIDATION SUMMARY ---")
print(f"Total checked: {total_checked}")
print(f"Valid images: {total_valid}")
print(f"Invalid images: {total_invalid}")
print(f"Report saved to: {report_path}")
