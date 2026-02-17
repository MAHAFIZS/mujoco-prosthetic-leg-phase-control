from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class UCIInertial:
    X: np.ndarray        # [N, T, 6] (acc xyz + gyro xyz)
    y: np.ndarray        # [N] original activity labels (1..6)


def _load_inertial_split(root: Path, split: str) -> UCIInertial:
    base = root / split
    sig = base / "Inertial Signals"

    # Use body acceleration + body gyro (cleaner than total_acc)
    acc_x = np.loadtxt(sig / f"body_acc_x_{split}.txt")
    acc_y = np.loadtxt(sig / f"body_acc_y_{split}.txt")
    acc_z = np.loadtxt(sig / f"body_acc_z_{split}.txt")

    gyr_x = np.loadtxt(sig / f"body_gyro_x_{split}.txt")
    gyr_y = np.loadtxt(sig / f"body_gyro_y_{split}.txt")
    gyr_z = np.loadtxt(sig / f"body_gyro_z_{split}.txt")

    # Stack to [N,T,6]
    X = np.stack([acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z], axis=-1)

    y = np.loadtxt(base / f"y_{split}.txt").astype(int).ravel()
    return UCIInertial(X=X, y=y)


def load_uci_inertial(root="data/uci_har/UCI HAR Dataset"):
    root = Path(root)
    tr = _load_inertial_split(root, "train")
    te = _load_inertial_split(root, "test")
    return tr, te
