from __future__ import annotations

from collections import Counter
from typing import Any

from sklearn.model_selection import train_test_split


def _can_stratify(labels: list[str]) -> bool:
    counts = Counter(labels)
    return len(counts) > 1 and min(counts.values()) >= 2


def assign_splits(
    records: list[dict[str, Any]],
    *,
    train_size: float = 0.8,
    dev_size: float = 0.1,
    test_size: float = 0.1,
    stratify_key: str = "topic",
    seed: int = 42,
) -> list[dict[str, Any]]:
    total = train_size + dev_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + dev_size + test_size must equal 1.0")
    if len(records) < 3:
        raise ValueError("Need at least 3 records to create train/dev/test splits")

    indices = list(range(len(records)))
    labels = [str(records[index].get(stratify_key, "")) for index in indices]
    stratify = labels if _can_stratify(labels) else None

    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    temp_dev_ratio = dev_size / (dev_size + test_size)
    temp_labels = [str(records[index].get(stratify_key, "")) for index in temp_indices]
    temp_stratify = temp_labels if _can_stratify(temp_labels) else None
    dev_indices, test_indices = train_test_split(
        temp_indices,
        train_size=temp_dev_ratio,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify,
    )

    split_by_index = {}
    split_by_index.update({index: "train" for index in train_indices})
    split_by_index.update({index: "dev" for index in dev_indices})
    split_by_index.update({index: "test" for index in test_indices})

    output = []
    for index, record in enumerate(records):
        with_split = dict(record)
        with_split["split"] = split_by_index[index]
        output.append(with_split)
    return output
