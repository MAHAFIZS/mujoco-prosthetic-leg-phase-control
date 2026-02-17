from __future__ import annotations
import numpy as np

def features_imu_windows(X: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Extract classic IMU features per window.
    X: [N, W, C] where C=2 here (left-foot, right-foot) but works for any C.

    Returns:
      F: [N, D]
      names: list of feature names length D
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X [N,W,C], got {X.shape}")

    N, W, C = X.shape

    mean = X.mean(axis=1)                 # [N,C]
    std = X.std(axis=1)                   # [N,C]
    p2p = X.max(axis=1) - X.min(axis=1)   # [N,C]
    energy = (X ** 2).sum(axis=1)         # [N,C]

    # Signal Magnitude Area (SMA) generalization:
    # sum over time of abs, normalized by window length
    sma = np.abs(X).sum(axis=1) / W       # [N,C]

    # Also include magnitude across channels (helps)
    mag = np.linalg.norm(X, axis=2)       # [N,W]
    mag_mean = mag.mean(axis=1)           # [N]
    mag_std = mag.std(axis=1)             # [N]
    mag_p2p = mag.max(axis=1) - mag.min(axis=1)  # [N]
    mag_energy = (mag ** 2).sum(axis=1)   # [N]

    feats = [mean, std, p2p, energy, sma,
             mag_mean[:, None], mag_std[:, None], mag_p2p[:, None], mag_energy[:, None]]
    F = np.concatenate(feats, axis=1)

    names = []
    for ci in range(C):
        names += [f"ch{ci}_mean", f"ch{ci}_std", f"ch{ci}_p2p", f"ch{ci}_energy", f"ch{ci}_sma"]
    names += ["mag_mean", "mag_std", "mag_p2p", "mag_energy"]

    return F.astype(np.float32), names
