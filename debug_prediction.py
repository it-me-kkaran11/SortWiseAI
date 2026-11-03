
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_DIR / "data" / "raw" / "trashnet" / "plastic" / "plastic16.jpg"
MODEL_PATH = PROJECT_DIR / "models" / "sortwise_mobilenetv3.pth"

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    class_names = checkpoint.get(
        "class_names",
        ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    )
else:
    state_dict = checkpoint
    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(class_names),
)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

image = Image.open(IMAGE_PATH).convert("RGB")
tensor = transform(image).unsqueeze(0)

with torch.inference_mode():
    probabilities = torch.softmax(model(tensor), dim=1)[0]

print(f"Image: {IMAGE_PATH.name}")
print("Expected dataset label: plastic")
print("\nModel predictions:")

for index in torch.argsort(probabilities, descending=True).tolist():
    print(f"{class_names[index]:12s} {probabilities[index].item():.2%}")

predicted_index = probabilities.argmax().item()
print(f"\nFinal prediction: {class_names[predicted_index]}")
