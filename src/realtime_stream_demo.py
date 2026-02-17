# src/realtime_stream_demo.py
from __future__ import annotations

import argparse
import time
import inspect
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from src.realtime_phase import stream_proxy_phases_from_uci


@dataclass
class LiveStats:
    last_phase: Optional[int] = None
    transitions: int = 0
    t0: Optional[float] = None

    def update(self, phase: int, t: float) -> None:
        if self.t0 is None:
            self.t0 = t
        if self.last_phase is None:
            self.last_phase = phase
            return
        if phase != self.last_phase:
            self.transitions += 1
            self.last_phase = phase

    def jitter_rate(self, t: float) -> float:
        if self.t0 is None:
            return 0.0
        dt = max(1e-9, t - self.t0)
        return self.transitions / dt


def _moving_transition_rate(ph: np.ndarray, dt: float) -> float:
    if ph.size < 2:
        return 0.0
    transitions = np.sum(ph[1:] != ph[:-1])
    total_t = (ph.size - 1) * dt
    return float(transitions / total_t) if total_t > 0 else 0.0


def _call_stream_filtered(**kwargs) -> Any:
    """
    Calls stream_proxy_phases_from_uci with only the kwargs it supports.
    Prevents crashes when function signature differs.
    """
    sig = inspect.signature(stream_proxy_phases_from_uci)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return stream_proxy_phases_from_uci(**filtered)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--window_sec", type=float, default=8.0)

    # You can keep this flag for UX, but your stream may not use it.
    p.add_argument("--phase_source", type=str, default="uci_walk_proxy_seq")

    p.add_argument("--latency_ms", type=float, default=0.0)
    p.add_argument("--uci_seed", type=int, default=0)
    p.add_argument("--mode", type=str, default="seq", choices=["seq", "random"])
    p.add_argument("--which", type=str, default="train", choices=["train", "test"])
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--no_sleep", action="store_true")
    args = p.parse_args()

    hz = float(args.hz)
    dt = 1.0 / hz
    n_steps = int(round(args.seconds * hz))
    maxlen = max(10, int(round(args.window_sec * hz)))

    # We prepare a superset of possible kwargs; _call_stream_filtered() will drop unsupported ones.
    gen = _call_stream_filtered(
        hz=hz,
        n_steps=n_steps,
        which=args.which,
        start_index=args.start_index,
        mode=args.mode,
        seed=args.uci_seed,
        latency_ms=args.latency_ms,
        sleep=(not args.no_sleep),
        # these may or may not exist in your generator — safe to include:
        phase_source=args.phase_source,
        uci_seed=args.uci_seed,
        no_sleep=args.no_sleep,
    )

    t_buf: Deque[float] = deque(maxlen=maxlen)
    conf_buf: Deque[float] = deque(maxlen=maxlen)
    raw_buf: Deque[int] = deque(maxlen=maxlen)
    applied_buf: Deque[int] = deque(maxlen=maxlen)

    stats = LiveStats()

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"Realtime phase stream — mode={args.mode}, hz={hz:.1f}, latency={args.latency_ms:.1f}ms"
    )

    ax1.set_ylabel("phase (0=swing, 1=stance)")
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("confidence")
    ax2.set_xlabel("time [s]")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)

    (line_applied,) = ax1.plot([], [], linewidth=2, label="phase (applied)")
    (line_raw,) = ax1.plot([], [], linestyle="--", linewidth=1.5, label="phase (raw)")
    (line_conf,) = ax2.plot([], [], linewidth=2, label="confidence")

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    txt = ax1.text(
        0.02, 0.85, "",
        transform=ax1.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    last_print = time.time()

    for st in gen:
        t = float(getattr(st, "t", 0.0))
        phase = int(getattr(st, "phase", 0))
        conf = float(getattr(st, "confidence", 0.5))

        # If your PhaseState doesn’t expose raw/applied separately, we mirror phase into both.
        raw_phase = getattr(st, "raw_phase", phase)
        applied_phase = getattr(st, "applied_phase", phase)

        t_buf.append(t)
        conf_buf.append(conf)
        raw_buf.append(int(raw_phase))
        applied_buf.append(int(applied_phase))

        stats.update(int(applied_phase), t)

        tt = np.asarray(t_buf)
        ph_app = np.asarray(applied_buf)
        ph_raw = np.asarray(raw_buf)
        cf = np.asarray(conf_buf)

        line_applied.set_data(tt, ph_app)
        line_raw.set_data(tt, ph_raw)
        line_conf.set_data(tt, cf)

        ax2.set_xlim(max(0.0, t - args.window_sec), max(args.window_sec, t))

        jitter_total = stats.jitter_rate(t)
        jitter_win = _moving_transition_rate(ph_app, dt)

        txt.set_text(
            f"t={t:6.2f}s\n"
            f"applied={int(applied_phase)} raw={int(raw_phase)}  conf={conf:0.3f}\n"
            f"jitter(total)={jitter_total:0.3f}/s\n"
            f"jitter(window)={jitter_win:0.3f}/s\n"
            f"latency={args.latency_ms:0.1f} ms"
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

        now = time.time()
        if now - last_print > 1.0:
            last_print = now
            print(
                f"[t={t:6.2f}s] applied={int(applied_phase)} raw={int(raw_phase)} "
                f"conf={conf:0.3f} jitter_total={jitter_total:0.3f}/s jitter_window={jitter_win:0.3f}/s"
            )

    print("Done.")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
