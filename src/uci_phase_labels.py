from __future__ import annotations
import numpy as np


def stance_swing_proxy(X: np.ndarray,
                       low_factor: float = 0.95,
                       high_factor: float = 1.05,
                       min_duration: int = 8):
    """
    X: [N,T,6]
    Returns:
        phase: [N,T]  (0=swing, 1=stance)
        thr: base threshold
    """

    acc = X[..., 0:3]
    gyr = X[..., 3:6]

    acc_mag = np.linalg.norm(acc, axis=-1)
    gyr_mag = np.linalg.norm(gyr, axis=-1)

    energy = acc_mag + 0.5 * gyr_mag

    # smooth
    k = 7
    kernel = np.ones(k) / k
    energy_s = np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="same"), 1, energy
    )

    thr = float(np.median(energy_s))

    low_thr = thr * low_factor
    high_thr = thr * high_factor

    N, T = energy_s.shape
    phase = np.zeros((N, T), dtype=int)

    for n in range(N):
        state = 1  # start in stance
        count = 0

        for t in range(T):
            e = energy_s[n, t]

            if state == 1:
                # stance → swing only if clearly high
                if e > high_thr:
                    state = 0
                    count = 0
            else:
                # swing → stance only if clearly low
                if e < low_thr:
                    state = 1
                    count = 0

            phase[n, t] = state
            count += 1

        # remove very short segments
        phase[n] = _remove_short_segments(phase[n], min_duration)

    return phase, thr


def _remove_short_segments(signal: np.ndarray, min_len: int):
    """Remove segments shorter than min_len."""
    out = signal.copy()
    T = len(signal)

    start = 0
    for i in range(1, T):
        if signal[i] != signal[i-1]:
            if i - start < min_len:
                out[start:i] = signal[i]
            start = i

    return out
