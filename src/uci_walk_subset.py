from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.uci_loader import load_uci


WALK_LABELS = {1, 2, 3}  # walking, upstairs, downstairs


@dataclass
class UCISubset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def make_walk_vs_rest(root="data/uci_har/UCI HAR Dataset") -> UCISubset:
    d = load_uci(root)
    Xtr, ytr = d["X_train"], d["y_train"]
    Xte, yte = d["X_test"], d["y_test"]

    ytr_bin = np.isin(ytr, list(WALK_LABELS)).astype(int)  # 1=walk, 0=rest
    yte_bin = np.isin(yte, list(WALK_LABELS)).astype(int)

    return UCISubset(X_train=Xtr, y_train=ytr_bin, X_test=Xte, y_test=yte_bin)
