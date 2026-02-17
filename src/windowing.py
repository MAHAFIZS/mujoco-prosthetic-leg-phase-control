from __future__ import annotations
import numpy as np

def sliding_window_indices(n_samples: int, win_len: int, step: int):
    """
    Generate (start, end) indices for sliding windows.
    """
    starts = np.arange(0, n_samples - win_len + 1, step)
    for s in starts:
        yield s, s + win_len


def window_signal(x: np.ndarray, win_len: int, step: int):
    """
    Slice signal into windows.
    x: [T, C] or [T]
    Returns:
        windows: [N, win_len, C]
    """
    if x.ndim == 1:
        x = x[:, None]

    T, C = x.shape
    windows = []

    for s, e in sliding_window_indices(T, win_len, step):
        windows.append(x[s:e])

    return np.stack(windows, axis=0)
