from __future__ import annotations
import numpy as np

def window_labels_binary(contact: np.ndarray, win_len: int, step: int, threshold: float = 0.5):
    """
    Assign 0/1 label to each window using majority contact.
    """
    labels = []
    T = contact.size

    starts = np.arange(0, T - win_len + 1, step)

    for s in starts:
        w = contact[s:s + win_len]
        labels.append(1 if w.mean() >= threshold else 0)

    return np.array(labels, dtype=np.int8)
