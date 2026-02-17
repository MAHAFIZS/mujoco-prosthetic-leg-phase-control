from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import numpy as np

@dataclass
class PhaseLabels:
    contact: np.ndarray      # [T] 0/1 stance contact for chosen foot
    phase: np.ndarray        # [T] 0/1 (0=swing, 1=stance) same as contact for 2-class
    events_used: Dict[str, Any]

def _pick_two_main_event_symbols(symbol_counts: Dict[str, int]) -> List[str]:
    """
    Heuristic: pick two most frequent non-noise symbols for stance toggling.
    Many WFDB annotation sets include a noise marker like '~'.
    We'll exclude '~' from the toggling candidates.
    """
    items = [(s, n) for s, n in symbol_counts.items() if s != "~"]
    items.sort(key=lambda x: x[1], reverse=True)
    # choose top-2
    return [items[0][0], items[1][0]] if len(items) >= 2 else [items[0][0]]

def build_binary_contact_from_events(t: np.ndarray, events: Dict[str, Any], prefer_symbols: Tuple[str, str] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convert event stream -> binary contact signal using a simple toggle model.

    Assumption (works for many gait event annotations):
      - Two event types alternate: heel-strike and toe-off (or contact on/off).
      - So contact toggles at each event.
    We don't know which symbol = on/off yet, so we will:
      - pick two dominant symbols (excluding '~')
      - take their merged event times
      - toggle contact at each event
      - choose the polarity (start in stance vs swing) that yields more realistic duty factor (~50-80% stance)

    This is robust and dataset-agnostic.
    """
    sym = np.array(events["symbol"], dtype=object)
    ev_t = np.array(events["time_s"], dtype=float)

    # Count symbols
    uniq, cnt = np.unique(sym, return_counts=True)
    counts = {str(u): int(c) for u, c in zip(uniq, cnt)}

    if prefer_symbols is None:
        cand = _pick_two_main_event_symbols(counts)
        if len(cand) < 2:
            raise ValueError(f"Not enough event symbols to toggle contact. counts={counts}")
        s1, s2 = cand[0], cand[1]
    else:
        s1, s2 = prefer_symbols

    mask = (sym == s1) | (sym == s2)
    toggle_times = np.sort(ev_t[mask])

    # Build contact by toggling
    def make_contact(start_in_contact: int) -> np.ndarray:
        contact = np.zeros_like(t, dtype=np.int8)
        state = int(start_in_contact)
        j = 0
        for i in range(t.size):
            while j < toggle_times.size and t[i] >= toggle_times[j]:
                state = 1 - state
                j += 1
            contact[i] = state
        return contact

    c0 = make_contact(0)
    c1 = make_contact(1)

    # Choose polarity by stance fraction heuristic
    frac0 = float(c0.mean())
    frac1 = float(c1.mean())

    # Typical stance fraction in normal gait ~0.55-0.70, but can vary.
    target = 0.62
    pick = c0 if abs(frac0 - target) < abs(frac1 - target) else c1
    picked_start = 0 if pick is c0 else 1

    info = {
        "symbol_counts": counts,
        "toggle_symbols": (s1, s2),
        "n_toggle_events": int(toggle_times.size),
        "start_in_contact": picked_start,
        "stance_fraction": float(pick.mean()),
    }
    return pick.astype(np.int8), info

def label_record_two_class(t: np.ndarray, events: Dict[str, Any], prefer_symbols: Tuple[str, str] | None = None) -> PhaseLabels:
    contact, info = build_binary_contact_from_events(t, events, prefer_symbols=prefer_symbols)
    phase = contact.copy()  # 2-class: stance == contact
    return PhaseLabels(contact=contact, phase=phase, events_used=info)
