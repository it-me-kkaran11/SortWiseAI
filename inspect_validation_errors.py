
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "validation_predictions.csv"
OUTPUT = ROOT / "reports" / "trash_validation_errors.png"

if not REPORT.exists():
    raise FileNotFoundError(
        f"Validation report not found: {REPORT}"
    )

df = pd.read_csv(REPORT)

# Only inspect misclassified validation images whose actual class is trash.
errors = df[
    (df["actual_class"] == "trash")
    & (~df["correct"].astype(bool))
].copy()

if errors.empty:
    print("No misclassified trash validation images found.")
    raise SystemExit(0)

columns = 3
cell_width = 320
image_height = 230
label_height = 90
rows = (len(errors) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * (image_height + label_height)),
    "white",
)

draw = ImageDraw.Draw(sheet)

for position, (_, row) in enumerate(errors.iterrows()):
    relative_path = str(row["file"]).replace("\\", "/")
    image_path = ROOT / relative_path

    if not image_path.exists():
        print(f"Missing image: {image_path}")
        continue

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((cell_width - 16, image_height - 16))

    x = (position % columns) * cell_width
    y = (position // columns) * (image_height + label_height)

    image_x = x + (cell_width - image.width) // 2
    image_y = y + (image_height - image.height) // 2

    sheet.paste(image, (image_x, image_y))

    confidence = float(row["confidence_score"])

    caption = [
        Path(relative_path).name,
        f"Actual: {row['actual_class']}",
        f"Predicted: {row['predicted_class']}",
        f"Model score: {confidence:.1%}",
    ]

    for line_no, line in enumerate(caption):
        draw.text(
            (x + 8, y + image_height + 5 + line_no * 19),
            line,
            fill="black",
        )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT)

print(f"Images inspected: {len(errors)}")
print(f"Contact sheet saved to: {OUTPUT}")
