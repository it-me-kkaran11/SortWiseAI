
from transformers import pipeline

print("Loading the pretrained image-classification model...")
print("The first run may take a few minutes to download model files.")

classifier = pipeline(
    task="image-classification",
    model="microsoft/resnet-50",
    device=-1
)

test_image = (
    "https://huggingface.co/datasets/"
    "huggingface/documentation-images/resolve/main/"
    "pipeline-cat-chonk.jpeg"
)

print("\nAnalyzing the test image...")

predictions = classifier(test_image, top_k=5)

print("\nTop 5 predictions:")
for result in predictions:
    label = result["label"]
    confidence = result["score"] * 100
    print(f"{label}: {confidence:.2f}%")

print("\nMODEL TEST COMPLETED")
