from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np


@dataclass
class PhaseLabels:
    contact: np.ndarray
    phase: np.ndarray
    info: Dict[str, Any]


def _build_contact_from_pairs(
    t: np.ndarray,
    toggle_times: np.ndarray,
    start_in_contact: int,
) -> np.ndarray:
    """
    Toggle contact at each event time (piecewise constant).
    """
    contact = np.zeros_like(t, dtype=np.int8)
    state = int(start_in_contact)
    j = 0
    for i in range(t.size):
        while j < toggle_times.size and t[i] >= toggle_times[j]:
            state = 1 - state
            j += 1
        contact[i] = state
    return contact


def _jitter_score(contact: np.ndarray) -> int:
    """
    Counts number of transitions in contact (lower is better).
    """
    return int(np.sum(np.abs(np.diff(contact))))


def label_contact_from_events_eventpair(
    t: np.ndarray,
    events: Dict[str, Any],
    toggle_symbols: Tuple[str, str] = ("/", "Q"),
    stance_frac_range: Tuple[float, float] = (0.45, 0.75),
) -> PhaseLabels:
    """
    Robust stance/swing labeling using only event toggles:
      - collect toggle events using chosen symbols
      - try start_in_contact = 0 or 1
      - pick the one that yields realistic stance fraction AND minimal jitter

    This is stable and does not assume swing has higher IMU dynamics.
    """
    sym = np.array(events["symbol"], dtype=object)
    ev_t = np.array(events["time_s"], dtype=float)

    mask = (sym == toggle_symbols[0]) | (sym == toggle_symbols[1])
    toggle_times = np.sort(ev_t[mask])

    if toggle_times.size < 10:
        raise ValueError("Too few toggle events.")

    c0 = _build_contact_from_pairs(t, toggle_times, start_in_contact=0)
    c1 = _build_contact_from_pairs(t, toggle_times, start_in_contact=1)

    frac0 = float(c0.mean())
    frac1 = float(c1.mean())
    jit0 = _jitter_score(c0)
    jit1 = _jitter_score(c1)

    lo, hi = stance_frac_range
    def score(frac: float, jit: int) -> Tuple[int, float]:
        # primary: within stance frac range
        in_range = 1 if (lo <= frac <= hi) else 0
        # secondary: closeness to center
        center = (lo + hi) / 2
        dist = abs(frac - center)
        # tertiary: jitter (lower is better)
        return (in_range, -dist, -jit)

    s0 = score(frac0, jit0)
    s1 = score(frac1, jit1)

    contact = c0 if s0 > s1 else c1
    picked_start = 0 if contact is c0 else 1

    info = {
        "toggle_symbols": toggle_symbols,
        "n_toggle_events": int(toggle_times.size),
        "start_in_contact": picked_start,
        "stance_fraction": float(contact.mean()),
        "jitter_transitions": _jitter_score(contact),
        "candidate": {
            "start0": {"stance_fraction": frac0, "jitter": jit0},
            "start1": {"stance_fraction": frac1, "jitter": jit1},
        },
    }

    return PhaseLabels(contact=contact, phase=contact.copy(), info=info)
