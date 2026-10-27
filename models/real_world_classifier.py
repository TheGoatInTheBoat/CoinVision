
"""Evaluate a trained CoinVision classifier on real-world coin photos.

Run from the repository root:
    python models/evaluate_real_world.py --checkpoint models/coin_classifier_seed88.pth
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent

IMAGE_ROOT = (
    REPO_ROOT
    / "datasets"
    / "US Coins - Local"
    / "coins"
)

DEFAULT_CHECKPOINT = MODEL_DIR / "coin_classifier_seed99.pth"

OUTPUT_CSV = MODEL_DIR / "real_world_predictions.csv"
OUTPUT_JSON = MODEL_DIR / "real_world_metrics.json"

# Folder names mapped to the class labels used by the model.
FOLDER_LABELS = {
    "Pennies": "Penny",
    "Nickels": "Nickel",
    "Quarters": "Quarter",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class CoinCNN(nn.Module):
    """Must match the architecture used during training."""

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

def get_images() -> list[dict]:
    """Read coin labels and metadata from filenames."""

    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_ROOT}")

    # Expected format: n1-rev-2018-mint=P.jpg
    import re

    pattern = re.compile(
        r"^(?P<coin>[npq])(?P<number>\d+)-"
        r"(?P<side>obv|rev)-"
        r"(?P<year>\d{4})-"
        r"mint=(?P<mint>[A-Za-zX])$",
        re.IGNORECASE,
    )

    denomination_map = {
        "n": "Nickel",
        "p": "Penny",
        "q": "Quarter",
    }

    images = []

    for folder_name in ("Pennies", "Nickels", "Quarters"):
        folder = IMAGE_ROOT / folder_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            match = pattern.fullmatch(path.stem)
            if not match:
                print(f"WARNING: Skipping unrecognized filename: {path.name}")
                continue

            data = match.groupdict()
            denomination = denomination_map[data["coin"].lower()]

            # Check that the filename's denomination agrees with its folder.
            expected_folder = {
                "Penny": "Pennies",
                "Nickel": "Nickels",
                "Quarter": "Quarters",
            }[denomination]

            if folder_name != expected_folder:
                raise ValueError(
                    f"Denomination mismatch: {path.name} is in {folder_name}"
                )

            images.append({
                "path": path,
                "actual": denomination,
                "coin_id": f"{data['coin'].lower()}{data['number']}",
                "side": data["side"].lower(),
                "year": int(data["year"]),
                "mint": data["mint"].upper(),
            })

    if not images:
        raise RuntimeError(f"No correctly named images found in {IMAGE_ROOT}")

    return images

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CoinVision on real-world coin images."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained .pth checkpoint.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}\n"
            "Pass the correct path using --checkpoint."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the saved model, not a newly initialized model.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("\n=== CHECKPOINT METADATA ===")
    print("Checkpoint keys:", checkpoint.keys())
    print("Saved classes:", checkpoint.get("classes"))
    print("Saved image size:", checkpoint.get("image_size"))
    
    classes = checkpoint.get(
        "classes", ["Penny", "Nickel", "Quarter"]
    )
    image_size = int(checkpoint.get("image_size", 512))

    model = CoinCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Match the original evaluation preprocessing.
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    images = get_images()

    print("CoinVision real-world evaluation")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training image size: {image_size} x {image_size}")
    print(f"Image directory: {IMAGE_ROOT}")
    print(f"Images found: {len(images)}")

    actual_indices = []
    predicted_indices = []
    csv_rows = []

    with torch.inference_mode():
        for item in images:
            image_path = item["path"]
            actual_label = item["actual"]
            if actual_label not in classes:
                raise ValueError(
                    f"Label {actual_label!r} is missing from "
                    f"checkpoint classes: {classes}"
                )

            try:
                with Image.open(image_path) as image:
                    tensor = transform(image.convert("RGB"))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                ) from exc

            logits = model(tensor.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)[0]

            predicted_index = int(probabilities.argmax().item())
            predicted_label = classes[predicted_index]
            confidence = float(probabilities[predicted_index].item())

            actual_index = classes.index(actual_label)
            actual_indices.append(actual_index)
            predicted_indices.append(predicted_index)

            row = {
                "image": str(image_path.relative_to(IMAGE_ROOT)),
                "coin_id": item["coin_id"],
                "side": item["side"],
                "year": item["year"],
                "mint": item["mint"],
                "actual": actual_label,
                "predicted": predicted_label,
                "confidence": round(confidence, 6),
                "correct": predicted_label == actual_label,
            }

            # Record each class probability for error analysis.
            for i, class_name in enumerate(classes):
                row[f"probability_{class_name.lower()}"] = round(
                    float(probabilities[i].item()), 6
                )

            csv_rows.append(row)

            status = "CORRECT" if row["correct"] else "WRONG"
            print(
                f"{status:7} | Actual: {actual_label:7} | "
                f"Predicted: {predicted_label:7} | "
                f"Confidence: {confidence:.1%} | "
                f"{row['image']}"
            )

    # Confusion matrix: rows = actual, columns = predicted.
    matrix = [
        [0 for _ in classes]
        for _ in classes
    ]

    for actual, predicted in zip(actual_indices, predicted_indices):
        matrix[actual][predicted] += 1

    total = len(actual_indices)
    correct_count = sum(
        actual == predicted
        for actual, predicted in zip(actual_indices, predicted_indices)
    )

    per_class = {}
    for i, class_name in enumerate(classes):
        support = sum(matrix[i])
        true_positives = matrix[i][i]
        predicted_count = sum(row[i] for row in matrix)

        per_class[class_name] = {
            "correct": true_positives,
            "total": support,
            "accuracy": (
                true_positives / support if support else None
            ),
            "precision": (
                true_positives / predicted_count
                if predicted_count else None
            ),
        }

    # Save detailed image-by-image predictions.
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(csv_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    report = {
        "checkpoint": str(checkpoint_path),
        "image_size": image_size,
        "classes": classes,
        "total_images": total,
        "correct_predictions": correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "confusion_matrix_rows_actual_columns_predicted": matrix,
        "per_class": per_class,
        "note": (
            "Obverse and reverse images from the same physical coin "
            "are paired observations, not independent coins."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== REAL-WORLD RESULTS ===")
    print(f"Correct: {correct_count}/{total}")
    print(f"Accuracy: {correct_count / total:.2%}")
    print("\nPer-denomination results:")

    for name, metrics in per_class.items():
        accuracy = metrics["accuracy"]
        accuracy_text = (
            f"{accuracy:.2%}" if accuracy is not None else "N/A"
        )
        print(
            f"  {name:7}: {metrics['correct']}/"
            f"{metrics['total']} ({accuracy_text})"
        )

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("             " + "  ".join(f"{name:>8}" for name in classes))
    for name, row in zip(classes, matrix):
        print(f"{name:>12} " + "  ".join(f"{value:8d}" for value in row))

    print(f"\nPredictions saved to: {OUTPUT_CSV}")
    print(f"Metrics saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
