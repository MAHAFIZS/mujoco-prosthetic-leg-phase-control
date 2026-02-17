from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

from src.uci_walk_subset import make_walk_vs_rest


def main():
    data = make_walk_vs_rest()
    X_train, y_train = data.X_train, data.y_train
    X_test, y_test = data.X_test, data.y_test

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=20000)),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== UCI HAR: Walk vs Rest (0=rest, 1=walk) ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(clf, out_dir / "uci_walk_vs_rest.joblib")

    metrics = {
        "task": "uci_walk_vs_rest",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "walk_frac_train": float(y_train.mean()),
        "walk_frac_test": float(y_test.mean()),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    (out_dir / "uci_walk_vs_rest_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved model: results/uci_walk_vs_rest.joblib")
    print("Saved metrics: results/uci_walk_vs_rest_metrics.json")


if __name__ == "__main__":
    main()
