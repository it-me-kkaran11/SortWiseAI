
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "reports" / "dataset_split.csv"

MODEL_PATHS = {
    "Original": PROJECT_DIR / "models" / "sortwise_mobilenetv3.pth",
    "Fine-tuned": PROJECT_DIR / "models" / "sortwise_mobilenetv3_finetuned.pth",
}

IMAGE_SIZE = 224
BATCH_SIZE = 32
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(6)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class WasteDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)
        self.class_to_index = {
            name: index for index, name in enumerate(CLASSES)
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = PROJECT_DIR / Path(
            str(row["file"]).replace("\\", "/")
        )

        with Image.open(path) as image:
            image = image.convert("RGB")
            image = transform(image)

        label = self.class_to_index[str(row["class"]).strip()]
        return image, label, str(row["file"])


def load_model(model_path):
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        classes = checkpoint.get("classes", CLASSES)
    else:
        state_dict = checkpoint
        classes = CLASSES

    if list(classes) != CLASSES:
        raise ValueError(
            f"Class order mismatch in {model_path}: {classes}"
        )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes),
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def evaluate(model, loader, model_name):
    total = 0
    correct = 0
    rows = []

    with torch.inference_mode():
        for images, labels, paths in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            probabilities = torch.softmax(model(images), dim=1)
            predictions = probabilities.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for i in range(len(paths)):
                rows.append({
                    "model": model_name,
                    "file": paths[i],
                    "actual": CLASSES[labels[i].item()],
                    "predicted": CLASSES[predictions[i].item()],
                    "confidence_score": probabilities[i, predictions[i]].item(),
                    "correct": predictions[i].item() == labels[i].item(),
                })

    accuracy = correct / total
    return accuracy, rows


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    dataframe = pd.read_csv(MANIFEST_PATH)
    test_df = dataframe[dataframe["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("No test images found in the split manifest.")

    dataset = WasteDataset(test_df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Device: {DEVICE}")
    print(f"Test images: {len(dataset)}")
    print("The test split is used only for evaluation.\n")

    summaries = []
    all_predictions = []

    for name, path in MODEL_PATHS.items():
        if not path.exists():
            print(f"SKIPPING {name}: checkpoint not found at {path}")
            continue

        print(f"Evaluating {name} model...")
        model = load_model(path)
        accuracy, predictions = evaluate(model, loader, name)

        correct = sum(row["correct"] for row in predictions)
        summaries.append({
            "model": name,
            "test_images": len(dataset),
            "correct": correct,
            "incorrect": len(dataset) - correct,
            "test_accuracy": accuracy,
        })
        all_predictions.extend(predictions)

        print(
            f"{name}: {correct}/{len(dataset)} correct "
            f"({accuracy:.2%})\n"
        )

        del model

    if not summaries:
        raise FileNotFoundError("No model checkpoints were found.")

    REPORT_DIR = PROJECT_DIR / "reports"
    pd.DataFrame(summaries).to_csv(
        REPORT_DIR / "model_comparison_summary.csv",
        index=False,
    )
    pd.DataFrame(all_predictions).to_csv(
        REPORT_DIR / "model_comparison_predictions.csv",
        index=False,
    )

    print("=" * 55)
    print("FINAL TEST COMPARISON")
    print("=" * 55)
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nReports saved:")
    print(REPORT_DIR / "model_comparison_summary.csv")
    print(REPORT_DIR / "model_comparison_predictions.csv")


if __name__ == "__main__":
    main()
