from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

from src.io_dataset import load_record
from src.labels import label_record_two_class
from src.windowing import window_signal
from src.window_labels import window_labels_binary
from src.features_imu import features_imu_windows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=str, default="als1")
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = load_record(args.id)
    lab = label_record_two_class(d.t, d.events)

    fs = int(round(d.fs))
    win_len = int(0.2 * fs)
    step = int(0.05 * fs)

    Xw = window_signal(d.imu, win_len, step)
    y = window_labels_binary(lab.contact, win_len, step)

    F, feat_names = features_imu_windows(Xw)

    # Simple train/test split for now (Day 4+ we do subject-independent splits)
    X_train, X_test, y_train, y_test = train_test_split(
        F, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)

    print("\n=== Classification report (0=swing, 1=stance) ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save artifacts
    dump({"scaler": scaler, "clf": clf, "feat_names": feat_names}, outdir / f"svm_{args.id}.joblib")

    metrics = {
        "record_id": args.id,
        "n_windows": int(F.shape[0]),
        "stance_frac": float(y.mean()),
        "features_dim": int(F.shape[1]),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, digits=4, output_dict=True),
        "label_info": lab.events_used,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved model to: {outdir / f'svm_{args.id}.joblib'}")
    print(f"Saved metrics to: {outdir / 'metrics.json'}")

if __name__ == "__main__":
    main()
