"""
Dataloader and train/val/held-out splits.
Held-out split is never touched during training — reserved for final evaluation only.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AntibodyEntry:
    pdb_id: str
    sequence: str
    label: float | None = None  # aggregation score; None = unlabeled


def load_pdb_ids(path: str = "data/antibody_pdb_ids.json") -> list[str]:
    with open(path) as f:
        return json.load(f)


def split_dataset(
    entries: list[AntibodyEntry],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split into train / val / held-out.
    Held-out is never used during training — only for final spot-checking.
    """
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    held_out = shuffled[val_end:]

    return train, val, held_out
