from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import wfdb


@dataclass
class UnifiedGaitData:
    t: np.ndarray                # [T] seconds
    imu: np.ndarray              # [T, C] raw IMU channels from WFDB record (cleaned)
    imu_channels: list[str]      # channel names
    fs: float                    # sampling rate Hz
    events: Dict[str, Any]       # raw annotation info from .ts
    meta: Dict[str, Any]


def _dataset_root() -> Path:
    return Path("data/public/gait-in-neurodegenerative-disease-database-1.0.0")


def _fill_nan_linear(x: np.ndarray) -> np.ndarray:
    """
    Fill NaN/Inf in each channel by linear interpolation over time, edge-filled.
    x: [T,C] or [T]
    """
    if x.ndim == 1:
        x = x[:, None]

    x = x.astype(float, copy=True)
    T, C = x.shape
    idx = np.arange(T)

    for c in range(C):
        col = x[:, c]

        # convert inf -> nan
        col = np.where(np.isfinite(col), col, np.nan)

        nans = np.isnan(col)
        if not nans.any():
            x[:, c] = col
            continue

        good = ~nans
        if good.sum() < 2:
            # too few points: replace with zeros
            x[:, c] = 0.0
            continue

        col[nans] = np.interp(idx[nans], idx[good], col[good])
        x[:, c] = col

    return x


def load_record(record_id: str, root: Optional[str | Path] = None) -> UnifiedGaitData:
    """
    Loads one WFDB record (e.g. 'als1', 'control3', 'park10', 'hunt7').

    Reads:
      - signals via wfdb.rdrecord()
      - events via wfdb.rdann(..., 'ts')
    """
    root_path = Path(root) if root is not None else _dataset_root()
    base = root_path / record_id

    rec = wfdb.rdrecord(str(base))
    fs = float(rec.fs)

    # signals
    sig = rec.p_signal if rec.p_signal is not None else rec.d_signal.astype(float)
    sig = _fill_nan_linear(sig)

    if not np.isfinite(sig).all():
        raise ValueError("Signal still contains NaN/Inf after cleaning.")

    sig_names = list(rec.sig_name) if rec.sig_name is not None else [f"ch{i}" for i in range(sig.shape[1])]
    t = np.arange(sig.shape[0], dtype=float) / fs

    # annotations (.ts)
    ann = wfdb.rdann(str(base), "ts")
    events = {
        "sample": ann.sample.copy(),
        "time_s": ann.sample.astype(float) / fs,
        "symbol": list(ann.symbol),
        "aux_note": list(ann.aux_note) if ann.aux_note is not None else [],
        "fs": fs,
    }

    meta = {
        "record_id": record_id,
        "root": str(root_path),
        "n_samples": int(sig.shape[0]),
        "n_channels": int(sig.shape[1]),
    }

    return UnifiedGaitData(t=t, imu=sig, imu_channels=sig_names, fs=fs, events=events, meta=meta)


def summarize_events(events: Dict[str, Any]) -> Dict[str, Any]:
    """Quick summary: unique symbols, aux_note examples, count, etc."""
    symbols = events.get("symbol", [])
    aux = events.get("aux_note", [])
    out: Dict[str, Any] = {}
    out["n_events"] = len(symbols)
    out["unique_symbols"] = sorted(set(symbols))
    if aux:
        out["unique_aux_note"] = sorted(set(a for a in aux if a))
        out["aux_note_examples"] = [a for a in aux if a][:10]
    else:
        out["unique_aux_note"] = []
        out["aux_note_examples"] = []
    return out
