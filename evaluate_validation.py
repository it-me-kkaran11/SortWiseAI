
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "reports" / "dataset_split.csv"
MODEL_PATH = ROOT / "models" / "sortwise_mobilenetv3_finetuned.pth"
REPORT_DIR = ROOT / "reports"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
torch.set_num_threads(6)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ValidationDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        self.class_to_index = {
            name: index for index, name in enumerate(CLASSES)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        relative_path = str(row["file"]).replace("\\", "/")
        image_path = ROOT / relative_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        label = str(row["class"]).strip().lower()
        if label not in self.class_to_index:
            raise ValueError(f"Unknown class: {label}")

        with Image.open(image_path) as image:
            image = TRANSFORM(image.convert("RGB"))

        return image, self.class_to_index[label], relative_path


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    dataframe = pd.read_csv(MANIFEST)
    validation_df = dataframe[
        dataframe["split"].astype(str).str.lower().str.strip()
        == "validation"
    ].copy()

    if validation_df.empty:
        raise ValueError("No validation images found.")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint_classes = checkpoint.get("classes", CLASSES)
    if list(checkpoint_classes) != CLASSES:
        raise ValueError(f"Class order mismatch: {checkpoint_classes}")

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(CLASSES)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    dataset = ValidationDataset(validation_df)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    rows = []
    total = 0
    correct = 0

    print("SORTWISE AI - VALIDATION ERROR ANALYSIS")
    print(f"Device: {DEVICE}")
    print(f"Validation images: {len(dataset)}")
    print("Evaluating without training or modifying the model...")

    with torch.inference_mode():
        for images, labels, paths in loader:
            probabilities = torch.softmax(
                model(images.to(DEVICE)), dim=1
            )
            predictions = probabilities.argmax(dim=1)
            scores = probabilities.max(dim=1).values

            for i in range(len(labels)):
                actual = int(labels[i].item())
                predicted = int(predictions[i].item())
                is_correct = actual == predicted

                confusion[actual][predicted] += 1
                total += 1
                correct += int(is_correct)

                rows.append({
                    "file": paths[i],
                    "actual_class": CLASSES[actual],
                    "predicted_class": CLASSES[predicted],
                    "confidence_score": round(
                        float(scores[i].item()), 6
                    ),
                    "correct": is_correct,
                })

    metrics = []
    for i, class_name in enumerate(CLASSES):
        tp = confusion[i][i]
        actual_count = sum(confusion[i])
        predicted_count = sum(
            confusion[r][i] for r in range(len(CLASSES))
        )

        precision = tp / predicted_count if predicted_count else 0
        recall = tp / actual_count if actual_count else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        metrics.append({
            "class": class_name,
            "validation_images": actual_count,
            "correct": tp,
            "incorrect": actual_count - tp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(rows)
    metrics_df = pd.DataFrame(metrics)
    confusion_df = pd.DataFrame(
        confusion, index=CLASSES, columns=CLASSES
    )
    confusion_df.index.name = "actual_class"

    predictions_df.to_csv(
        REPORT_DIR / "validation_predictions.csv", index=False
    )
    metrics_df.to_csv(
        REPORT_DIR / "validation_class_metrics.csv", index=False
    )
    confusion_df.to_csv(
        REPORT_DIR / "validation_confusion_matrix.csv"
    )

    print("\nVALIDATION RESULTS")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {correct / total:.2%}")
    print("\nPER-CLASS METRICS")
    print(metrics_df.to_string(index=False))
    print("\nCONFUSION MATRIX")
    print("Rows = actual; columns = predicted")
    print(confusion_df.to_string())
    print("\nReports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
