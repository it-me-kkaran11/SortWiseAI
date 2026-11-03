
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ==================================================
# SORTWISE AI - FINE-TUNED MODEL EVALUATION
# ==================================================

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "reports" / "dataset_split.csv"

MODEL_PATH = (
    ROOT / "models" / "sortwise_mobilenetv3_finetuned.pth"
)

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
torch.set_num_threads(6)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class TestDataset(Dataset):
    def __init__(self, dataframe, classes):
        self.df = dataframe.reset_index(drop=True)
        self.class_to_index = {
            name: index for index, name in enumerate(classes)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        relative_path = str(row["file"]).replace("\\", "/")
        image_path = ROOT / relative_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"Test image not found: {image_path}"
            )

        label = str(row["class"]).strip().lower()

        if label not in self.class_to_index:
            raise ValueError(f"Unexpected class: {label}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = IMAGE_TRANSFORM(image)

        return (
            image,
            self.class_to_index[label],
            relative_path,
        )


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Unexpected checkpoint format.")

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain model_state_dict."
        )

    classes = checkpoint.get("classes", EXPECTED_CLASSES)

    if list(classes) != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class order: {classes}"
        )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, classes


def main():
    print("=" * 60)
    print("SORTWISE AI - FINE-TUNED MODEL TEST EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH.name}")

    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST}"
        )

    dataframe = pd.read_csv(MANIFEST)

    required_columns = {"file", "class", "split"}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Manifest requires {required_columns}; "
            f"found {set(dataframe.columns)}"
        )

    dataframe["split"] = (
        dataframe["split"].astype(str).str.lower().str.strip()
    )
    dataframe["class"] = (
        dataframe["class"].astype(str).str.lower().str.strip()
    )

    # Use only the pre-existing held-out test split.
    test_df = dataframe[dataframe["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("No test images found.")

    model, classes = load_model()

    unknown_labels = set(test_df["class"]) - set(classes)
    if unknown_labels:
        raise ValueError(
            f"Unknown labels in test set: {unknown_labels}"
        )

    dataset = TestDataset(test_df, classes)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    confusion = [
        [0 for _ in classes] for _ in classes
    ]

    prediction_rows = []
    total = 0
    correct_total = 0
    start_time = time.time()

    print(f"Test images: {len(dataset)}")
    print("Evaluating model. Please wait...")

    with torch.inference_mode():
        for batch_number, (images, labels, paths) in enumerate(
            loader, start=1
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)

            for i in range(len(labels)):
                actual_index = int(labels[i].item())
                predicted_index = int(predicted[i].item())
                score = float(confidence[i].item())

                confusion[actual_index][predicted_index] += 1

                is_correct = actual_index == predicted_index
                correct_total += int(is_correct)
                total += 1

                prediction_rows.append({
                    "file": paths[i],
                    "actual_class": classes[actual_index],
                    "predicted_class": classes[predicted_index],
                    "confidence_score": round(score, 6),
                    "correct": is_correct,
                })

            processed = min(
                batch_number * BATCH_SIZE, len(dataset)
            )
            print(
                f"\rProcessed {processed}/{len(dataset)} images",
                end="",
                flush=True,
            )

    elapsed = time.time() - start_time
    accuracy = correct_total / total if total else 0.0

    # Calculate per-class metrics.
    metric_rows = []

    for index, class_name in enumerate(classes):
        true_positive = confusion[index][index]
        actual_count = sum(confusion[index])
        predicted_count = sum(
            confusion[row][index] for row in range(len(classes))
        )

        precision = (
            true_positive / predicted_count
            if predicted_count else 0.0
        )
        recall = (
            true_positive / actual_count
            if actual_count else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        metric_rows.append({
            "class": class_name,
            "test_images": actual_count,
            "correct": true_positive,
            "incorrect": actual_count - true_positive,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
        })

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    confusion_df = pd.DataFrame(
        confusion,
        index=classes,
        columns=classes,
    )
    confusion_df.index.name = "actual_class"

    # Macro averages give each class equal weight.
    macro_precision = metrics_df["precision"].mean()
    macro_recall = metrics_df["recall"].mean()
    macro_f1 = metrics_df["f1_score"].mean()

    summary_df = pd.DataFrame([{
        "model": "Fine-tuned MobileNetV3-Small",
        "test_images": total,
        "correct_predictions": correct_total,
        "incorrect_predictions": total - correct_total,
        "test_accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1_score": round(macro_f1, 4),
        "elapsed_seconds": round(elapsed, 2),
    }])

    # Preserve existing report names.
    metrics_df.to_csv(
        REPORT_DIR / "test_class_metrics.csv", index=False
    )
    predictions_df.to_csv(
        REPORT_DIR / "test_predictions.csv", index=False
    )
    confusion_df.to_csv(
        REPORT_DIR / "test_confusion_matrix.csv"
    )
    summary_df.to_csv(
        REPORT_DIR / "test_summary.csv", index=False
    )

    print("\n\n" + "=" * 60)
    print("FINE-TUNED MODEL - FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test images: {total}")
    print(f"Correct: {correct_total}")
    print(f"Incorrect: {total - correct_total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro precision: {macro_precision:.2%}")
    print(f"Macro recall: {macro_recall:.2%}")
    print(f"Macro F1-score: {macro_f1:.2%}")
    print(f"Evaluation time: {elapsed:.1f} seconds")

    print("\nPER-CLASS METRICS")
    print(metrics_df.to_string(index=False))

    print("\nCONFUSION MATRIX")
    print("Rows = actual class; columns = predicted class")
    print(confusion_df.to_string())

    print("\nReports saved in:", REPORT_DIR)
    print(
        "\nNote: softmax scores are not calibrated probabilities "
        "and do not guarantee correctness."
    )


if __name__ == "__main__":
    main()
