
from pathlib import Path
import copy
import random
import time

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ============================================================
# SORTWISE AI - MOBILE NET V3 PARTIAL FINE-TUNING
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "reports" / "dataset_split.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "sortwise_mobilenetv3_finetuned.pth"
HISTORY_PATH = REPORT_DIR / "finetuned_training_history.csv"

CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 10
PATIENCE = 3
CPU_THREADS = 6

HEAD_LR = 0.0003
FEATURE_LR = 0.00003
WEIGHT_DECAY = 0.0001


# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

torch.set_num_threads(CPU_THREADS)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

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

        image_path = PROJECT_DIR / Path(
            str(row["filepath"]).replace("\\", "/")
        )

        label = self.class_to_index[str(row["label"])]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
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


# ============================================================
# TRAINING / VALIDATION LOOP
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()

        # Keep frozen pretrained feature layers in evaluation mode.
        model.features.eval()

        # Re-enable training mode only for the unfrozen final blocks.
        for layer in model.features[-3:]:
            layer.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_images += batch_size

        if training and batch_index % 15 == 0:
            print(
                f"  Batch {batch_index}/{len(loader)} "
                f"| {total_images}/{len(loader.dataset)} images",
                flush=True,
            )

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("SORTWISE AI - PARTIAL FINE-TUNING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"New model output: {MODEL_PATH}")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    # Read the existing split; do not generate a new split.
    dataframe = pd.read_csv(MANIFEST_PATH)

    required_columns = {"file", "class", "split"}
    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Expected manifest columns {sorted(required_columns)}, "
            f"but found {dataframe.columns.tolist()}"
        )

    dataframe = dataframe.rename(
        columns={"file": "filepath", "class": "label"}
    )

    dataframe["label"] = dataframe["label"].astype(str).str.strip()
    dataframe["split"] = dataframe["split"].astype(str).str.strip()

    if set(dataframe["label"].unique()) != set(CLASSES):
        raise ValueError(
            "Manifest classes do not match the expected six classes: "
            f"{sorted(dataframe['label'].unique())}"
        )

    train_df = dataframe[dataframe["split"] == "train"].copy()
    val_df = dataframe[dataframe["split"] == "validation"].copy()
    test_df = dataframe[dataframe["split"] == "test"].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "Train, validation and test splits must all be non-empty."
        )

    # Verify that all manifest paths exist before training.
    for _, row in dataframe.iterrows():
        path = PROJECT_DIR / Path(
            str(row["filepath"]).replace("\\", "/")
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"Image referenced by manifest was not found: {path}"
            )

    print(f"\nImages in manifest: {len(dataframe)}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Test images reserved: {len(test_df)}")
    print("\nTraining class counts:")
    print(train_df["label"].value_counts().reindex(CLASSES).to_string())

    train_dataset = WasteDataset(train_df, train_transform)
    val_dataset = WasteDataset(val_df, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Start with pretrained MobileNetV3-Small weights.
    print("\nLoading pretrained MobileNetV3-Small weights...", flush=True)

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    # Freeze all feature extractor parameters first.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Fine-tune the final three feature blocks.
    for layer in model.features[-3:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    # Replace the final classification layer.
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        input_features,
        len(CLASSES),
    )

    model = model.to(DEVICE)

    # Use a lower learning rate for pretrained features.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for p in model.features.parameters()
                    if p.requires_grad
                ],
                "lr": FEATURE_LR,
            },
            {
                "params": model.classifier.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=0.000001,
    )

    criterion = nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    print("\nStarting fine-tuning...", flush=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        start_time = time.time()

        print(f"\nEpoch {epoch}/{MAX_EPOCHS}", flush=True)

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion
        )

        scheduler.step(val_accuracy)

        elapsed = time.time() - start_time

        print(f"Train loss: {train_loss:.4f}", flush=True)
        print(f"Train accuracy: {train_accuracy:.2%}", flush=True)
        print(f"Validation loss: {val_loss:.4f}", flush=True)
        print(f"Validation accuracy: {val_accuracy:.2%}", flush=True)
        print(f"Epoch time: {elapsed:.1f} seconds", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "epoch_seconds": elapsed,
        })

        # Save only the best checkpoint according to validation accuracy.
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": best_state,
                "classes": CLASSES,
                "image_size": IMAGE_SIZE,
                "architecture": "mobilenet_v3_small",
                "best_validation_accuracy": best_accuracy,
                "seed": SEED,
            }, MODEL_PATH)

            print(f"Best model saved: {MODEL_PATH}", flush=True)
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement "
                f"({epochs_without_improvement}/{PATIENCE})",
                flush=True,
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping: validation accuracy did not improve.")
            break

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    print("\n" + "=" * 60)
    print("FINE-TUNING FINISHED")
    print("=" * 60)
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print(f"Fine-tuned model: {MODEL_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print("\nThe test set was not used for training or model selection.")
    print("Next, evaluate the saved model on the held-out test set.")


if __name__ == "__main__":
    main()
