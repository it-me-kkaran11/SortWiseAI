
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "reports" / "dataset_split.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 5
IMAGE_SIZE = 224
NUM_WORKERS = 0
CPU_THREADS = 6

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# 2. REPRODUCIBILITY
# ==================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==================================================
# 3. DATASET
# ==================================================

class WasteDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.class_to_index = {
            name: index for index, name in enumerate(CLASSES)
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(str(row["filepath"]))
        label = str(row["label"]).strip().lower()

        if not image_path.is_absolute():
            image_path = ROOT / image_path

        if label not in self.class_to_index:
            raise ValueError(f"Unexpected class label: {label}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.class_to_index[label]


def load_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Cannot find {MANIFEST}. Run prepare_dataset.py first."
        )

    df = pd.read_csv(MANIFEST)

    # Adapt the existing CSV columns to the names used by this script.
    df = df.rename(columns={
        "file": "filepath",
        "class": "label",
    })

    required = {"filepath", "label", "split"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    unexpected = set(df["label"].unique()) - set(CLASSES)
    if unexpected:
        raise ValueError(f"Unexpected labels: {sorted(unexpected)}")

    for split in ("train", "validation", "test"):
        if not (df["split"] == split).any():
            raise ValueError(f"No images found for split: {split}")

    missing_files = []

    for path in df["filepath"].astype(str):
        candidate = Path(path)

        if not candidate.is_absolute():
            candidate = ROOT / candidate

        if not candidate.is_file():
            missing_files.append(str(candidate))

            if len(missing_files) >= 10:
                break

    if missing_files:
        raise FileNotFoundError(
            "Some image paths in the manifest do not exist. "
            "First missing paths:\n" + "\n".join(missing_files)
        )

    return df


# ==================================================
# 4. IMAGE TRANSFORMS
# ==================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==================================================
# 5. TRAINING AND EVALUATION
# ==================================================

def run_epoch(model, loader, loss_function, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0
    start_time = time.time()

    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(
            loader, start=1
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total_images += batch_size

            # Show progress during training.
            if training and (
                batch_index == 1
                or batch_index % 10 == 0
                or batch_index == len(loader)
            ):
                elapsed = time.time() - start_time
                print(
                    f"\rTraining batch {batch_index}/{len(loader)} "
                    f"| images {total_images}/{len(loader.dataset)} "
                    f"| elapsed {elapsed / 60:.1f} min",
                    end="",
                    flush=True,
                )

    if training:
        print()

    return (
        total_loss / total_images,
        total_correct / total_images,
    )


# ==================================================
# 6. MAIN TRAINING PROCESS
# ==================================================

def main():
    print("=" * 60, flush=True)
    print("SORTWISE AI - WASTE CLASSIFIER TRAINING", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"Torchvision: {torchvision.__version__}", flush=True)

    if DEVICE.type == "cpu":
        torch.set_num_threads(CPU_THREADS)
        print(f"CPU threads: {torch.get_num_threads()}", flush=True)

    df = load_manifest()

    print(f"\nImages in manifest: {len(df)}", flush=True)
    print("\nSplit counts:", flush=True)
    print(df["split"].value_counts().to_string(), flush=True)
    print("\nClass counts:", flush=True)
    print(df.groupby(["split", "label"]).size().to_string(), flush=True)

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    train_dataset = WasteDataset(train_df, train_transform)
    val_dataset = WasteDataset(val_df, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("\nLoading pretrained ResNet-50 weights...", flush=True)

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Freeze pretrained feature-extraction layers.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the final layer for our six waste categories.
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(CLASSES))
    model = model.to(DEVICE)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=0.001,
    )

    best_val_accuracy = -1.0
    best_state = None
    history = []

    print("\nStarting training...", flush=True)
    print(
        f"Training images: {len(train_dataset)} | "
        f"Validation images: {len(val_dataset)} | "
        f"Batches per epoch: {len(train_loader)}",
        flush=True,
    )
    print("The test split will not be used during training.", flush=True)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{EPOCHS}", flush=True)

        train_loss, train_accuracy = run_epoch(
            model, train_loader, loss_function, optimizer
        )

        val_loss, val_accuracy = run_epoch(
            model, val_loader, loss_function
        )

        epoch_seconds = time.time() - epoch_start

        result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "epoch_seconds": round(epoch_seconds, 2),
        }
        history.append(result)

        print(
            f"Epoch {epoch}/{EPOCHS} complete | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.2%} | "
            f"validation loss: {val_loss:.4f} | "
            f"validation accuracy: {val_accuracy:.2%} | "
            f"time: {epoch_seconds / 60:.1f} min",
            flush=True,
        )

        # Save the best checkpoint immediately so progress is preserved.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

            model_path = MODEL_DIR / "sortwise_resnet50.pth"
            torch.save({
                "model_state_dict": best_state,
                "classes": CLASSES,
                "image_size": IMAGE_SIZE,
                "architecture": "resnet50",
                "seed": SEED,
                "best_validation_accuracy": best_val_accuracy,
            }, model_path)

            print(
                f"Best model checkpoint saved: {model_path}",
                flush=True,
            )

        # Update the history file after every epoch.
        history_path = REPORT_DIR / "training_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)

    print("\n" + "=" * 60, flush=True)
    print("TRAINING COMPLETED", flush=True)
    print(f"Best validation accuracy: {best_val_accuracy:.2%}", flush=True)
    print(
        f"Saved model: {MODEL_DIR / 'sortwise_resnet50.pth'}",
        flush=True,
    )
    print(
        f"Saved training history: {REPORT_DIR / 'training_history.csv'}",
        flush=True,
    )
    print(
        "The held-out test set remains untouched. "
        "Next, we will evaluate it separately.",
        flush=True,
    )


if __name__ == "__main__":
    main()
