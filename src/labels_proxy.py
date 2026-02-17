from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import butter, filtfilt


@dataclass
class PhaseLabels:
    contact: np.ndarray
    phase: np.ndarray
    info: Dict[str, Any]


def _lowpass(x: np.ndarray, fs: float, cutoff_hz: float = 6.0, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, x)


def label_contact_with_proxy(
    t: np.ndarray,
    imu: np.ndarray,
    events: Dict[str, Any],
    toggle_symbols: Tuple[str, str] = ("/", "Q"),
) -> PhaseLabels:
    """
    Build stance/swing labels using:
      - event times to segment intervals
      - IMU magnitude proxy to decide which intervals are stance

    Output: contact[t] 1=stance, 0=swing
    """
    fs = float(events.get("fs", 1.0 / np.mean(np.diff(t))))

    sym = np.array(events["symbol"], dtype=object)
    ev_t = np.array(events["time_s"], dtype=float)

    mask = (sym == toggle_symbols[0]) | (sym == toggle_symbols[1])
    toggle_times = np.sort(ev_t[mask])

    # IMU magnitude proxy
    imu = np.nan_to_num(imu, nan=0.0, posinf=0.0, neginf=0.0)
    mag = np.linalg.norm(imu, axis=1)
    mag_lp = _lowpass(mag, fs, cutoff_hz=6.0)

    # interval score = mean absolute derivative (swing tends to be "more dynamic")
    dmag = np.abs(np.diff(mag_lp, prepend=mag_lp[0]))

    contact = np.zeros_like(t, dtype=np.int8)

    # For each interval [ti, ti+1), classify stance vs swing by motion level
    interval_scores = []
    interval_bounds = []

    for i in range(len(toggle_times) - 1):
        a = toggle_times[i]
        b = toggle_times[i + 1]
        ia = int(np.searchsorted(t, a, side="left"))
        ib = int(np.searchsorted(t, b, side="left"))
        if ib <= ia + 5:
            continue
        score = float(dmag[ia:ib].mean())
        interval_scores.append(score)
        interval_bounds.append((ia, ib))

    if len(interval_scores) < 10:
        raise ValueError("Too few intervals; cannot label reliably.")

    scores = np.array(interval_scores)
    thr = np.median(scores)  # dynamic intervals above median -> swing (0), below -> stance (1)

    for (ia, ib), s in zip(interval_bounds, scores):
        is_stance = 1 if s <= thr else 0
        contact[ia:ib] = is_stance

    info = {
        "toggle_symbols": toggle_symbols,
        "n_toggle_events": int(toggle_times.size),
        "n_intervals": int(len(interval_bounds)),
        "threshold": float(thr),
        "stance_fraction": float(contact.mean()),
    }
    return PhaseLabels(contact=contact, phase=contact.copy(), info=info)
