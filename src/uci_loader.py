# src/uci_loader.py
from __future__ import annotations
from pathlib import Path
import numpy as np


def load_uci_raw_windows(root: str | Path = "data/uci_har/UCI HAR Dataset"):
    root = Path(root)

    # Inertial Signals files (already segmented windows)
    def read_split(split: str):
        base = root / split / "Inertial Signals"
        signals = []
        for name in [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
        ]:
            f = base / f"{name}_{split}.txt"
            arr = np.loadtxt(f)  # [N,128]
            signals.append(arr[..., None])  # [N,128,1]
        X = np.concatenate(signals, axis=2)  # [N,128,6]
        return X

    X_train = read_split("train")
    X_test = read_split("test")
    return X_train, X_test
