from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Dict, List
import time

import numpy as np


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class PhaseState:
    phase: int              # 0=swing, 1=stance
    confidence: float       # confidence proxy (0..1)
    t: float                # timestamp (sec)


# -----------------------------
# UCI HAR raw loader + labels
# -----------------------------
def _default_uci_root() -> Path:
    candidates = [
        Path("data/uci_har/UCI HAR Dataset"),
        Path("data/uci_har/UCI HAR Dataset/UCI HAR Dataset"),
        Path("data/uci_har") / "UCI HAR Dataset",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in Path("data").rglob("UCI HAR Dataset"):
        if p.is_dir():
            return p
    raise FileNotFoundError("Could not find 'UCI HAR Dataset' under ./data. Check your path.")


def _load_inertial_set(root: Path, split: str) -> np.ndarray:
    """
    Returns X_windows: [N,128,6]
    Channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    """
    assert split in ("train", "test")
    base = root / split / "Inertial Signals"

    def read_txt(name: str) -> np.ndarray:
        return np.loadtxt(str(base / name), dtype=np.float32)

    ax = read_txt(f"body_acc_x_{split}.txt")
    ay = read_txt(f"body_acc_y_{split}.txt")
    az = read_txt(f"body_acc_z_{split}.txt")
    gx = read_txt(f"body_gyro_x_{split}.txt")
    gy = read_txt(f"body_gyro_y_{split}.txt")
    gz = read_txt(f"body_gyro_z_{split}.txt")

    X = np.stack([ax, ay, az, gx, gy, gz], axis=-1)  # [N,128,6]
    return X


def _load_labels(root: Path, split: str) -> np.ndarray:
    assert split in ("train", "test")
    return np.loadtxt(str(root / split / f"y_{split}.txt"), dtype=np.int64)


def load_uci_raw_windows_with_labels(
    root: Optional[str | Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rootp = Path(root) if root is not None else _default_uci_root()
    Xtr = _load_inertial_set(rootp, "train")
    Xte = _load_inertial_set(rootp, "test")
    ytr = _load_labels(rootp, "train")
    yte = _load_labels(rootp, "test")
    return Xtr, Xte, ytr, yte


def _walking_mask(y: np.ndarray) -> np.ndarray:
    # 1 WALKING, 2 UPSTAIRS, 3 DOWNSTAIRS
    return np.isin(y, [1, 2, 3])


# -----------------------------
# Proxy phase inside a window
# -----------------------------
def stance_swing_proxy_window(Xwin: np.ndarray, thr: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Compute per-sample stance proxy for ONE window.
    Xwin: [128,6] (acc xyz, gyro xyz)
    Returns:
      phase_seq: [128] int {0,1}
      thr_used: float
    """
    Xwin = np.asarray(Xwin, dtype=np.float32)
    if Xwin.ndim != 2 or Xwin.shape[1] != 6:
        raise ValueError(f"Expected [T,6], got {Xwin.shape}")

    acc = Xwin[:, 0:3]
    gyro = Xwin[:, 3:6]

    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    # simple motion magnitude
    m = acc_mag + 0.5 * gyro_mag

    if thr is None:
        # robust threshold: median + 0.5*mad
        med = float(np.median(m))
        mad = float(np.median(np.abs(m - med)) + 1e-8)
        thr_used = med + 0.5 * mad
    else:
        thr_used = float(thr)

    # interpret "low motion" as stance (1), "high motion" as swing (0)
    # (You can flip this if you want the opposite convention.)
    phase = (m < thr_used).astype(np.int32)

    return phase, thr_used


def _debounce_binary(seq: np.ndarray, min_len: int = 3) -> np.ndarray:
    """
    Remove very short runs to reduce flicker.
    min_len in samples.
    """
    seq = seq.astype(np.int32).copy()
    if len(seq) == 0:
        return seq

    start = 0
    cur = seq[0]
    for i in range(1, len(seq) + 1):
        if i == len(seq) or seq[i] != cur:
            run_len = i - start
            if run_len < min_len:
                # flip short run to neighbor value if possible
                left = seq[start - 1] if start > 0 else None
                right = seq[i] if i < len(seq) else None
                if left is not None:
                    seq[start:i] = left
                elif right is not None:
                    seq[start:i] = right
            if i < len(seq):
                start = i
                cur = seq[i]
    return seq


# -----------------------------
# Streaming proxy phases at hz
# -----------------------------
def stream_proxy_phases_from_uci(
    hz: float = 20.0,
    n_steps: int = 400,
    which: str = "train",          # train/test/all
    start_index: int = 0,
    mode: str = "sequential",      # sequential/random
    seed: int = 0,
    sleep: bool = True,
    walk_only: bool = False,
    debounce_samples: int = 3,
) -> Iterator[PhaseState]:
    """
    Outputs a REAL toggling stance/swing signal (per-sample proxy inside each UCI window).

    - Each UCI window: 128 samples @ 50 Hz => 2.56 s
    - We compute proxy phase per sample (50 Hz), then resample to `hz`
    """
    Xtr, Xte, ytr, yte = load_uci_raw_windows_with_labels()

    if which == "train":
        Xw, y = Xtr, ytr
    elif which == "test":
        Xw, y = Xte, yte
    else:
        Xw = np.concatenate([Xtr, Xte], axis=0)
        y = np.concatenate([ytr, yte], axis=0)

    if walk_only:
        m = _walking_mask(y)
        Xw = Xw[m]
        if len(Xw) == 0:
            raise RuntimeError("walk_only=True but no walking windows found.")

    rng = np.random.default_rng(int(seed))

    dt = 1.0 / float(hz)
    fs = 50.0
    T = Xw.shape[1]  # 128
    win_sec = T / fs  # 2.56

    # how many output ticks per window at hz
    out_per_win = max(1, int(round(win_sec * hz)))

    step = 0
    idx = int(start_index)

    while step < n_steps:
        if mode == "random":
            widx = int(rng.integers(0, len(Xw)))
        else:
            widx = idx % len(Xw)

        Xwin = Xw[widx]  # [128,6]

        phase_50, thr = stance_swing_proxy_window(Xwin, thr=None)
        if debounce_samples > 0:
            phase_50 = _debounce_binary(phase_50, min_len=int(debounce_samples))

        # resample 50Hz -> hz by picking indices
        # indices in [0..127]
        pick = np.linspace(0, T - 1, out_per_win).round().astype(int)
        phase_hz = phase_50[pick]

        # confidence: distance from threshold (scaled)
        # (bigger margin => higher confidence)
        acc = Xwin[:, 0:3]
        gyro = Xwin[:, 3:6]
        m_sig = np.linalg.norm(acc, axis=1) + 0.5 * np.linalg.norm(gyro, axis=1)
        margin = np.abs(m_sig[pick] - thr)
        conf = 1.0 / (1.0 + np.exp(-margin))  # logistic

        for j in range(len(phase_hz)):
            if step >= n_steps:
                break
            yield PhaseState(phase=int(phase_hz[j]), confidence=float(conf[j]), t=step * dt)
            if sleep:
                time.sleep(dt)
            step += 1

        idx += 1
