from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "datasets" / "US Coins - Kaggle"
METADATA_PATH = DATASET_ROOT / "metadata.csv"

SEED = 77

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def create_split(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    """Create a seeded, 70/15/15 split. Denominations remain proportional"""

    rng = random.Random(seed)

    rows_by_denomination: dict[str, list[dict[str, str]]] = {}

    for row in rows:
        denomination = row["denomination"]
        rows_by_denomination.setdefault(denomination, []).append(row)

    output_rows = []

    for denomination_rows in rows_by_denomination.values():
        rng.shuffle(denomination_rows)

        total = len(denomination_rows)

        train_count = round(total * TRAIN_RATIO)
        validation_count = round(total * VALIDATION_RATIO)
        test_count = total - train_count - validation_count

        assignments = (
            [("train", row) for row in denomination_rows[:train_count]]
            + [
                ("validation", row)
                for row in denomination_rows[
                    train_count:train_count + validation_count
                ]
            ]
            + [
                ("test", row)
                for row in denomination_rows[
                    train_count + validation_count:
                ]
            ]
        )

        for split, row in assignments:
            output_rows.append(
                {
                    "image_id": row["image_id"],
                    "split": split,
                }
            )

    rng.shuffle(output_rows)

    return output_rows


def main() -> None:
    with METADATA_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as metadata_file:
        reader = csv.DictReader(metadata_file)
        metadata_rows = list(reader)

    split_rows = create_split(metadata_rows, SEED)

    output_path = DATASET_ROOT / f"split - seed={SEED}.csv"

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["image_id", "split"],
        )
        writer.writeheader()
        writer.writerows(split_rows)

    counts = Counter(row["split"] for row in split_rows)

    print(f"Created: {output_path}")
    print(f"Seed: {SEED}")
    print(f"Total images: {len(split_rows):,}")
    print(f"Training: {counts['train']:,}")
    print(f"Validation: {counts['validation']:,}")
    print(f"Testing: {counts['test']:,}")


if __name__ == "__main__":
    main()
