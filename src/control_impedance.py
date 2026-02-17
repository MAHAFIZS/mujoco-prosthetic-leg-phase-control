from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ImpedanceGains:
    """Impedance gains for 2 joints: knee, ankle."""
    K_knee: float
    D_knee: float
    K_ankle: float
    D_ankle: float


DEFAULT_GAINS: Dict[int, ImpedanceGains] = {
    # 0 = SWING  (more compliant)
    0: ImpedanceGains(K_knee=20.0, D_knee=2.0,  K_ankle=10.0, D_ankle=1.0),
    # 1 = STANCE (stiffer / more damping for support)
    1: ImpedanceGains(K_knee=80.0, D_knee=8.0,  K_ankle=50.0, D_ankle=5.0),
}


class PhaseImpedanceController:
    """
    Phase-dependent impedance:
        tau = K(phase) * (q_ref - q) - D(phase) * qd

    Assumes:
      q, qd, q_ref are length-2 arrays ordered [knee, ankle]
    """

    def __init__(
        self,
        gains: Dict[int, ImpedanceGains] | None = None,
        torque_limit: float = 150.0,
        tau_rate_limit: float = 500.0,  # Nm/s (simple smoothing limit)
    ):
        self.gains = gains if gains is not None else DEFAULT_GAINS
        self.torque_limit = float(torque_limit)
        self.tau_rate_limit = float(tau_rate_limit)

        self._tau_prev = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self._tau_prev[:] = 0.0

    def get_gains(self, phase: int) -> ImpedanceGains:
        if phase not in self.gains:
            # fallback to stance if unknown
            return self.gains[1]
        return self.gains[phase]

    def step(
        self,
        phase: int,
        q: np.ndarray,
        qd: np.ndarray,
        q_ref: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32).reshape(2)
        qd = np.asarray(qd, dtype=np.float32).reshape(2)
        q_ref = np.asarray(q_ref, dtype=np.float32).reshape(2)
        dt = float(dt)

        g = self.get_gains(int(phase))

        K = np.array([g.K_knee, g.K_ankle], dtype=np.float32)
        D = np.array([g.D_knee, g.D_ankle], dtype=np.float32)

        tau = K * (q_ref - q) - D * qd

        # torque saturation
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)

        # simple torque rate limiting (smoothness)
        if dt > 0:
            max_step = self.tau_rate_limit * dt
            tau = np.clip(tau, self._tau_prev - max_step, self._tau_prev + max_step)

        self._tau_prev = tau
        return tau


def torque_derivative_rms(tau: np.ndarray, dt: float) -> float:
    """
    tau: [T,2] torque sequence
    Returns RMS of d(tau)/dt across time and joints (a smoothness metric).
    """
    tau = np.asarray(tau, dtype=np.float32)
    dt = float(dt)
    if tau.ndim != 2 or tau.shape[1] != 2 or tau.shape[0] < 2 or dt <= 0:
        return float("nan")
    d = np.diff(tau, axis=0) / dt
    return float(np.sqrt(np.mean(d ** 2)))


def slip_distance_xy(foot_xy: np.ndarray) -> float:
    """
    Very simple 'foot slip' proxy: total path length in XY.
    foot_xy: [T,2]
    """
    foot_xy = np.asarray(foot_xy, dtype=np.float32)
    if foot_xy.ndim != 2 or foot_xy.shape[1] != 2 or foot_xy.shape[0] < 2:
        return float("nan")
    d = np.diff(foot_xy, axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))
