"""CoinVision CNN baseline: classify coin images as Penny, Nickel, or Quarter.

Dependencies: pip install torch torchvision pillow
Run: python models/image_classifier.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 88

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "datasets" / "US Coins - Kaggle"
METADATA_PATH = DATASET_ROOT / "metadata.csv"
SPLITS_PATH = DATASET_ROOT / str("split - seed=" + str(SEED) + ".csv")
IMAGE_ROOT = DATASET_ROOT / "coins"
MODEL_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = MODEL_DIR / str("coin_classifier_seed" + str(SEED) + ".pth")
REPORT_PATH = MODEL_DIR / str("coin_classifier_seed" + str(SEED) + "_metrics.json")


IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # Safe default for Windows.
CLASSES = ["Penny", "Nickel", "Quarter"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
RESIZED_CATEGORY_DIRS = {
    "Jefferson Nickels, 1938-Date": "Jefferson Nickels, 1938-Date SIZE=512",
    "Lincoln Cents, 1909-Date": "Lincoln Cents, 1909-Date SIZE=512",
    "Washington Quarters, 1932-1998": "Washington Quarters, 1932-1998 SIZE=512",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(metadata_path: str) -> Path:
    """Prefer the resized SIZE=512 image, falling back to the original path."""
    relative = Path(metadata_path.replace("\\", "/"))
    parts = relative.parts
    if len(parts) >= 3 and parts[0].lower() == "coins":
        category = parts[1]
        resized_category = RESIZED_CATEGORY_DIRS.get(category, f"{category} SIZE=512")
        candidate = IMAGE_ROOT / resized_category
        for part in parts[2:]:
            candidate /= part
        if candidate.exists():
            return candidate
    return DATASET_ROOT / relative


class CoinDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform: transforms.Compose):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = resolve_image_path(row["image_path"])
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASS_TO_INDEX[row["denomination"].strip()]


class CoinCNN(nn.Module):
    """Small CNN trained from scratch as an initial baseline."""
    def __init__(self, num_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def load_splits() -> dict[str, list[dict[str, str]]]:
    metadata = read_csv(METADATA_PATH)
    split_rows = read_csv(SPLITS_PATH)
    split_by_id = {r["image_id"].strip(): r["split"].strip().lower() for r in split_rows}
    result = {name: [] for name in ("train", "validation", "test")}
    for row in metadata:
        split = split_by_id.get(row["image_id"].strip())
        denomination = row["denomination"].strip()
        if split in result and denomination in CLASS_TO_INDEX:
            result[split].append(row)
    return result


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    evaluation = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return train, evaluation


def make_loaders(rows: dict[str, list[dict[str, str]]]) -> dict[str, DataLoader]:
    train_transform, eval_transform = make_transforms()
    datasets = {
        "train": CoinDataset(rows["train"], train_transform),
        "validation": CoinDataset(rows["validation"], eval_transform),
        "test": CoinDataset(rows["test"], eval_transform),
    }
    generator = torch.Generator().manual_seed(SEED)
    return {
        "train": DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, generator=generator),
        "validation": DataLoader(datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
        "test": DataLoader(datasets["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    targets_all, predictions_all = [], []
    with (torch.enable_grad() if training else torch.no_grad()):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            predictions = logits.argmax(dim=1)
            n = targets.size(0)
            loss_sum += loss.item() * n
            correct += (predictions == targets).sum().item()
            count += n
            targets_all.extend(targets.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())
    return loss_sum / max(count, 1), correct / max(count, 1), targets_all, predictions_all


def make_test_metrics(targets: list[int], predictions: list[int]) -> dict:
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    for actual, predicted in zip(targets, predictions):
        matrix[actual][predicted] += 1
    per_class = {}
    for i, name in enumerate(CLASSES):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted_count = sum(row[i] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    accuracy = sum(a == p for a, p in zip(targets, predictions)) / len(targets) if targets else 0.0
    return {
        "accuracy": accuracy,
        "class_order": CLASSES,
        "confusion_matrix_rows_actual_columns_predicted": matrix,
        "per_class": per_class,
        "num_examples": len(targets),
    }


def main() -> None:
    set_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_splits()
    print("CoinVision CNN baseline")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Splits: {SPLITS_PATH}")
    for name in ("train", "validation", "test"):
        counts = Counter(row["denomination"].strip() for row in rows[name])
        print(f"{name.capitalize():10s}: {len(rows[name]):,} images | {dict(counts)}")

    loaders = make_loaders(rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoinCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_val_accuracy = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, _, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, loaders["validation"], criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                        "validation_loss": val_loss, "validation_accuracy": val_acc})
        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f}, acc {train_acc:.4f} | val loss {val_loss:.4f}, acc {val_acc:.4f}")
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES,
                        "image_size": IMAGE_SIZE, "seed": SEED, "epoch": epoch,
                        "validation_accuracy": val_acc}, CHECKPOINT_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_targets, test_predictions = run_epoch(model, loaders["test"], criterion, device)
    metrics = make_test_metrics(test_targets, test_predictions)
    report = {"seed": SEED, "split_file": SPLITS_PATH.name, "classes": CLASSES,
              "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE, "epochs_requested": EPOCHS,
              "best_epoch": checkpoint["epoch"], "best_validation_accuracy": checkpoint["validation_accuracy"],
              "test_loss": test_loss, "test_accuracy": test_acc, "test_metrics": metrics,
              "training_history": history}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nFinal test results (best validation checkpoint)")
    print(f"Test loss: {test_loss:.4f}\nTest accuracy: {test_acc:.4f}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("             " + "  ".join(f"{name:>8}" for name in CLASSES))
    for name, row in zip(CLASSES, metrics["confusion_matrix_rows_actual_columns_predicted"]):
        print(f"{name:>12} " + "  ".join(f"{value:8d}" for value in row))
    print(f"\nSaved model: {CHECKPOINT_PATH}")
    print(f"Saved metrics: {REPORT_PATH}")


if __name__ == "__main__":
    main()
