from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

from src.io_dataset import load_record
from src.windowing import window_signal
from src.window_labels import window_labels_binary
from src.features_imu import features_imu_windows
from src.labels_eventpair import label_contact_from_events_eventpair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=str, default="als1")
    ap.add_argument("--outdir", type=str, default="results")
    ap.add_argument("--train_frac", type=float, default=0.7)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    d = load_record(args.id)
    lab = label_contact_from_events_eventpair(d.t, d.events, toggle_symbols=("/", "Q"))
    print("Label info:", lab.info)

    fs = int(round(d.fs))
    win_len = int(0.2 * fs)
    step = int(0.05 * fs)

    Xw = window_signal(d.imu, win_len, step)
    y = window_labels_binary(lab.contact, win_len, step)

    F, feat_names = features_imu_windows(Xw)

    N = F.shape[0]
    n_train = int(args.train_frac * N)
    X_train, y_train = F[:n_train], y[:n_train]
    X_test, y_test = F[n_train:], y[n_train:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=20000)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    print("\n=== EVENT-PAIR TIME-SPLIT report (0=swing, 1=stance) ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    dump({"scaler": scaler, "clf": clf, "feat_names": feat_names}, outdir / f"svm_eventpair_{args.id}.joblib")

    metrics = {
        "record_id": args.id,
        "split": "time",
        "train_frac": args.train_frac,
        "label_info": lab.info,
        "n_windows": int(N),
        "stance_frac_window": float(y.mean()),
        "features_dim": int(F.shape[1]),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, digits=4, output_dict=True),
    }
    (outdir / f"metrics_eventpair_{args.id}.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics to: {outdir / f'metrics_eventpair_{args.id}.json'}")


if __name__ == "__main__":
    main()
