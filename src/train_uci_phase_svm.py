# src/train_uci_phase_svm.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.uci_loader import load_uci_raw_windows   # windows in time-domain [N,T,6]
from src.uci_phase_labels import stance_swing_proxy


RESULTS = Path("results")


def flatten_features(Xw: np.ndarray) -> np.ndarray:
    """
    Quick baseline features from a window [T,6]:
    mean, std, energy (sum of squares), peak-to-peak per channel
    => 4 * 6 = 24 dims
    """
    mean = Xw.mean(axis=1)
    std = Xw.std(axis=1)
    energy = (Xw ** 2).sum(axis=1)
    ptp = np.ptp(Xw, axis=1)   # <-- FIXED

    return np.concatenate([mean, std, energy, ptp], axis=1)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_duration", type=int, default=8)
    args = ap.parse_args()

    # 1) Load raw windows (time domain)
    X_train_w, X_test_w = load_uci_raw_windows()
    Xw = np.concatenate([X_train_w, X_test_w], axis=0)  # [N,T,6]

    # 2) Create proxy phase per sample timestep -> convert to window label
    phase_seq, thr = stance_swing_proxy(Xw, min_duration=args.min_duration)  # [N,T]
    y = (phase_seq.mean(axis=1) > 0.5).astype(int)  # 0=swing, 1=stance (majority vote)

    # 3) Features
    X = flatten_features(Xw)

    # 4) Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    # 5) Train SVM
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=20000))
    ])
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)

    print("\n=== UCI HAR: Proxy STANCE vs SWING (0=swing, 1=stance) ===")
    print(classification_report(y_te, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_te, y_pred))

    RESULTS.mkdir(exist_ok=True, parents=True)
    model_path = RESULTS / "uci_proxy_phase_svm.joblib"
    metrics_path = RESULTS / "uci_proxy_phase_metrics.json"

    joblib.dump({"model": clf, "thr": thr, "min_duration": args.min_duration}, model_path)

    metrics = {
        "task": "uci_proxy_stance_vs_swing",
        "threshold": float(thr),
        "min_duration": int(args.min_duration),
        "n_samples": int(X.shape[0]),
        "stance_fraction": float(y.mean()),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
