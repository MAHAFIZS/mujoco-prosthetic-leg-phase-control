from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.control_impedance import PhaseImpedanceController, torque_derivative_rms
from src.realtime_phase import stream_proxy_phases_from_uci


def synthetic_phase(t: float, period_s: float = 1.0, stance_fraction: float = 0.62) -> int:
    tmod = (t % period_s) / period_s
    return 1 if tmod < stance_fraction else 0


def step_plant(q, qd, tau, dt):
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    tau = np.asarray(tau, dtype=float)

    M = np.array([1.0, 0.8])
    B = np.array([1.5, 1.0])
    K = np.array([4.0, 3.0])

    qdd = (tau - B * qd - K * q) / M
    qd_next = qd + qdd * dt
    q_next = q + qd_next * dt
    return q_next, qd_next


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_plots", action="store_true")
    ap.add_argument("--T", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--latency_ms", type=float, default=0.0)
    ap.add_argument("--no_sleep", action="store_true")

    ap.add_argument(
        "--phase_source",
        type=str,
        default="synthetic",
        choices=[
            "synthetic",
            "uci_proxy_seq",
            "uci_proxy_random",
            "uci_walk_proxy_seq",
            "uci_walk_proxy_random",
        ],
    )
    ap.add_argument("--uci_hz", type=float, default=20.0)
    ap.add_argument("--uci_seed", type=int, default=0)
    ap.add_argument("--uci_which", type=str, default="train", choices=["train", "test", "all"])
    ap.add_argument("--debounce_samples", type=int, default=3)

    args = ap.parse_args()

    T = float(args.T)
    dt = float(args.dt)
    steps = int(round(T / dt))

    latency_steps = int(round((args.latency_ms / 1000.0) / dt))
    latency_steps = max(0, latency_steps)

    ctrl = PhaseImpedanceController()

    q = np.array([0.2, 0.1], dtype=float)
    qd = np.array([0.0, 0.0], dtype=float)
    q_ref = np.array([0.0, 0.0], dtype=float)

    phase_source = args.phase_source
    phase_gen = None
    phase_dt = dt

    if phase_source == "synthetic":
        phase_dt = dt
    else:
        mode = "sequential" if "seq" in phase_source else "random"
        walk_only = "walk" in phase_source

        phase_dt = 1.0 / float(args.uci_hz)
        phase_gen = stream_proxy_phases_from_uci(
            hz=args.uci_hz,
            n_steps=int(np.ceil(T / phase_dt)) + 10,
            which=args.uci_which,
            start_index=0,
            mode=mode,
            seed=args.uci_seed,
            sleep=(not args.no_sleep),
            walk_only=walk_only,
            debounce_samples=args.debounce_samples,
        )

    buf = deque([0] * (latency_steps + 1), maxlen=(latency_steps + 1))

    ts = np.zeros(steps, dtype=float)
    phases_raw = np.zeros(steps, dtype=int)
    phases_applied = np.zeros(steps, dtype=int)
    qs = np.zeros((steps, 2), dtype=float)
    qds = np.zeros((steps, 2), dtype=float)
    taus = np.zeros((steps, 2), dtype=float)

    next_phase_update_t = 0.0
    current_raw_phase = 0

    for k in range(steps):
        t = k * dt
        ts[k] = t

        if phase_source == "synthetic":
            current_raw_phase = synthetic_phase(t, period_s=1.0, stance_fraction=0.62)
        else:
            if t >= next_phase_update_t:
                st = next(phase_gen)
                current_raw_phase = int(st.phase)
                next_phase_update_t += phase_dt

        phases_raw[k] = current_raw_phase

        buf.append(current_raw_phase)
        applied_phase = int(buf[0])
        phases_applied[k] = applied_phase

        tau = ctrl.step(applied_phase, q, qd, q_ref, dt)
        taus[k] = tau

        q, qd = step_plant(q, qd, tau, dt)
        qs[k] = q
        qds[k] = qd

    stance_fraction = float(phases_applied.mean())
    tau_smooth = float(torque_derivative_rms(taus, dt))

    print(f"Ran {T:.2f}s @ dt={dt:.4f} ({steps} steps)")
    print(f"Phase source: {phase_source}, latency_ms={args.latency_ms:.1f} (steps={latency_steps})")
    print(f"Stance fraction (applied): {stance_fraction:.3f}")
    print(f"Torque smoothness (RMS d(tau)/dt): {tau_smooth:.3f}")

    if args.save_plots:
        outdir = Path("results/figures")
        outdir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.step(ts, phases_applied, where="post")
        plt.ylim(-0.2, 1.2)
        plt.xlabel("time [s]")
        plt.ylabel("phase (0=swing, 1=stance)")
        plt.title(f"Gait phase (applied) — source={phase_source}")
        p1 = outdir / "day3_phase.png"
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p1}")

        plt.figure()
        plt.step(ts, phases_raw, where="post", label="raw")
        plt.step(ts, phases_applied, where="post", label="applied (delayed)")
        plt.ylim(-0.2, 1.2)
        plt.xlabel("time [s]")
        plt.ylabel("phase")
        plt.title(f"Raw vs applied phase (latency={args.latency_ms:.0f} ms)")
        plt.legend()
        p2 = outdir / "day3_phase_raw_vs_delayed.png"
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p2}")

        plt.figure()
        plt.plot(ts, qs[:, 0], label="q1")
        plt.plot(ts, qs[:, 1], label="q2")
        plt.xlabel("time [s]")
        plt.ylabel("angle [rad]")
        plt.title("Joint angles")
        plt.legend()
        p3 = outdir / "day3_angles.png"
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p3}")

        plt.figure()
        plt.plot(ts, taus[:, 0], label="tau1")
        plt.plot(ts, taus[:, 1], label="tau2")
        plt.xlabel("time [s]")
        plt.ylabel("torque [Nm]")
        plt.title(f"Torques (smoothness RMS d(tau)/dt={tau_smooth:.1f})")
        plt.legend()
        p4 = outdir / "day3_torques.png"
        plt.savefig(p4, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p4}")


if __name__ == "__main__":
    main()
