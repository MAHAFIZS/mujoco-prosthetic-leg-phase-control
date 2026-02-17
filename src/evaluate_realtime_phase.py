from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.realtime_phase import stream_phases_from_uci


def jitter_rate(phases: np.ndarray, hz: float) -> float:
    """Transitions per second."""
    changes = np.sum(phases[1:] != phases[:-1])
    return float(changes) / float(len(phases) / hz)


def main():
    hz = 20.0
    states = list(stream_phases_from_uci(hz=hz, n_steps=400, which="train", index=0))
    t = np.array([s.t for s in states])
    ph = np.array([s.phase for s in states], dtype=int)
    conf = np.array([s.confidence for s in states], dtype=float)

    jr = jitter_rate(ph, hz)
    print(f"Jitter rate: {jr:.3f} transitions/sec (~{jr*60:.1f} per minute)")

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(t, ph, label="phase (0=swing,1=stance)")
    plt.ylim(-0.2, 1.2)
    plt.title("Real-time phase stream (smoothed)")
    plt.xlabel("time (s)")
    plt.ylabel("phase")
    plt.legend()
    plt.savefig("results/figures/realtime_phase.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(t, conf, label="confidence")
    plt.title("Model confidence proxy")
    plt.xlabel("time (s)")
    plt.ylabel("conf")
    plt.legend()
    plt.savefig("results/figures/realtime_confidence.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("Saved plots to results/figures/")


if __name__ == "__main__":
    main()
