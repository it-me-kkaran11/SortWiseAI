# ♻️ SortWise AI
### AI-Powered Smart Waste Identification, Classification and Segregation System

<img width="1000" height="700" alt="Screenshot 2026-11-03 233705" src="https://github.com/user-attachments/assets/270c75ce-98c0-4e07-82ff-b1deee359d0e" />


**Problem Statement:** GP-05 — Smart Waste Segregation System

**Theme:** Sustainable Waste Management | Environmental Sustainability | AI/ML

---

## 1. Overview

SortWise AI is an AI-powered waste identification and segregation project designed to improve waste management through computer vision and machine learning.

The system analyses uploaded waste images and classifies them into different waste categories, helping users understand how waste can be appropriately segregated.

The initial prototype uses a fine-tuned MobileNetV3-Small image classification model trained on the TrashNet dataset. The application provides predictions, confidence scores, alternative classifications and general disposal recommendations through an interactive Streamlit interface.

The long-term vision is to evolve SortWise AI into a comprehensive waste analysis platform capable of identifying individual products, understanding their material composition, determining their environmental properties and recommending appropriate recycling, recovery and disposal methods.

## 2. Problem Statement

Improper waste segregation is a significant environmental challenge. Different materials require different treatment and disposal methods, but conventional waste segregation often depends on manual identification.

Existing waste classification approaches frequently focus on identifying broad categories such as paper, plastic, glass and metal. However, real-world waste products can contain multiple materials, making accurate classification and appropriate disposal more complicated.

For example:
- A snack wrapper may contain multiple layers of plastic and metallised material.
- A disposable cup may contain paper with a plastic lining.
- A used tissue may be paper-based but contaminated.
- Batteries require special collection and handling.
- Organic waste can be processed through composting or other biological treatment methods.

A single classification label cannot adequately describe all these properties.

SortWise AI aims to address this challenge by progressively developing a more detailed and intelligent waste identification and segregation system.

## 3. Project Objectives

- Develop an AI/ML-based waste image classification system.
- Identify common waste materials using computer vision.
- Improve classification performance through transfer learning and fine-tuning.
- Provide confidence scores and alternative predictions.
- Support informed waste segregation decisions.
- Expand towards identifying products containing multiple materials.
- Explore biodegradable and non-biodegradable waste classification.
- Recommend appropriate recycling, composting, recovery and disposal methods.

## 4. Current Implementation

The current prototype implements six waste categories:

| Category | Description |
|---|---|
| Cardboard | Cardboard-based waste |
| Glass | Glass-based waste |
| Metal | Metallic waste |
| Paper | Paper-based waste |
| Plastic | Plastic-based waste |
| Trash | General miscellaneous waste |

### Implemented Features

- Image upload through a Streamlit web interface.
- Image classification using MobileNetV3-Small.
- Transfer learning with fine-tuning.
- Prediction confidence scores.
- Top-three classification predictions.
- General waste disposal recommendations.
- Low-confidence prediction warnings.
- Model evaluation using a separate held-out test dataset.
- Comparison between original and fine-tuned models.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | Pretrained models and image transformations |
| MobileNetV3-Small | Image classification architecture |
| Pillow | Image processing |
| Pandas | Dataset management and evaluation |
| NumPy | Numerical operations |
| Streamlit | Web application interface |
| Git and
