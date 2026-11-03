
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# --------------------------------------------------
# SORTWISE AI - SMART WASTE CLASSIFICATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_DIR / "models" / "sortwise_mobilenetv3_finetuned.pth"
)

EXPECTED_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_SIZE = 224

CLASS_INFO = {
    "cardboard": {
        "title": "Cardboard",
        "bin": "Paper and cardboard recycling, where accepted",
        "advice": (
            "Keep cardboard dry. Remove excess tape and packaging "
            "where practical, and follow local recycling rules."
        ),
    },
    "glass": {
        "title": "Glass",
        "bin": "Glass collection, where accepted",
        "advice": (
            "Handle broken glass carefully. Check whether your local "
            "collection system accepts the specific glass item."
        ),
    },
    "metal": {
        "title": "Metal",
        "bin": "Metal recycling, where accepted",
        "advice": (
            "Empty containers and follow local cleaning requirements. "
            "Sharp objects, pressurized containers, and hazardous "
            "metal items may need special handling."
        ),
    },
    "paper": {
        "title": "Paper",
        "bin": "Paper recycling, where accepted",
        "advice": (
            "Keep paper clean and dry. Food-soiled paper may not be "
            "accepted in ordinary paper recycling."
        ),
    },
    "plastic": {
        "title": "Plastic",
        "bin": "Plastic collection, where accepted",
        "advice": (
            "This model recognizes only the broad plastic category. "
            "It does not identify the polymer type, resin code, or "
            "actual recyclable percentage. Check local acceptance rules."
        ),
    },
    "trash": {
        "title": "Other / General Waste",
        "bin": "Inspect the item before disposal",
        "advice": (
            "The model's trash category does not prove that an item "
            "belongs in general waste. Batteries, electronics, chemicals, "
            "medicines, sharps, and medical waste may require separate "
            "handling. Do not place potentially hazardous items in "
            "ordinary waste based on this prediction."
        ),
    },
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@st.cache_resource
def load_model():
    """Load and validate the fine-tuned model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found:\n{MODEL_PATH}\n"
            "Verify that training completed successfully."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("The model checkpoint has an unexpected format.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint.get("classes", EXPECTED_CLASSES)
    else:
        state_dict = checkpoint
        class_names = EXPECTED_CLASSES

    if list(class_names) != EXPECTED_CLASSES:
        raise ValueError(
            "The checkpoint class order does not match the app's "
            f"expected classes.\nCheckpoint: {class_names}"
        )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(class_names),
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names


# --------------------------------------------------
# PAGE DESIGN
# --------------------------------------------------

st.set_page_config(
    page_title="SortWise AI | Smart Waste Analyzer",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ SortWise AI")
st.subheader("Smart Waste Analyzer")

st.write(
    "Upload an image to classify waste into one of six broad "
    "categories and receive preliminary disposal guidance."
)

st.info(
    "Research prototype: the classifier can make mistakes. "
    "It does not yet identify specific plastic polymers, calculate "
    "recyclable percentages, or reliably detect all hazardous waste."
)

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded waste image",
            width="stretch",
        )

        if st.button(
            "Analyze Waste",
            type="primary",
            width="stretch",
        ):
            try:
                with st.spinner("Analyzing image with SortWise AI..."):
                    model, class_names = load_model()
                    input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

                    with torch.inference_mode():
                        outputs = model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)[0]

                    confidence, predicted_index = torch.max(
                        probabilities,
                        dim=0,
                    )

                    predicted_class = class_names[predicted_index.item()]
                    confidence_value = confidence.item()

                details = CLASS_INFO[predicted_class]

                st.divider()
                st.header("Analysis Result")

                st.metric(
                    label="Predicted Category",
                    value=details["title"],
                    delta=f"{confidence_value:.1%} model confidence score",
                )

                st.progress(confidence_value)

                # This is a provisional threshold, not a calibrated
                # guarantee of prediction reliability.
                if confidence_value < 0.60:
                    st.warning(
                        "Low model score. The prediction may be wrong. "
                        "Please inspect the item manually."
                    )
                else:
                    st.info(
                        "The model produced a prediction, but even a "
                        "high score does not guarantee correctness."
                    )

                st.subheader("Suggested Disposal Guidance")
                st.write(f"**General route:** {details['bin']}")
                st.write(details["advice"])

                st.subheader("Top 3 Predictions")

                top_count = min(3, len(class_names))
                top_scores, top_indices = torch.topk(
                    probabilities,
                    k=top_count,
                )

                for score, index in zip(top_scores, top_indices):
                    class_name = class_names[index.item()]
                    display_name = CLASS_INFO[class_name]["title"]

                    st.write(f"**{display_name}:** {score.item():.1%}")
                    st.progress(score.item())

                st.caption(
                    "Model: Fine-tuned MobileNetV3-Small | "
                    "Held-out test accuracy: 84.64% "
                    "(325 correct out of 384 images). "
                    "Real-world performance may differ."
                )

            except FileNotFoundError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.caption(
                    "Copy the full terminal error if you need help "
                    "troubleshooting."
                )

    except Exception as error:
        st.error(f"Could not open this image: {error}")

st.divider()
st.caption("SortWise AI | GP-05 Smart Waste Segregation Prototype")
