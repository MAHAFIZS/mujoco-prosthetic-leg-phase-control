from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.io_dataset import load_record, summarize_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=str, default="als1", help="record id like als1, control1, park1, hunt1")
    args = ap.parse_args()

    d = load_record(args.id)

    print("\n=== RECORD META ===")
    print(d.meta)
    print("\n=== IMU CHANNELS ===")
    for i, name in enumerate(d.imu_channels):
        print(f"{i:02d}: {name}")

    print("\n=== EVENTS SUMMARY (.ts) ===")
    s = summarize_events(d.events)
    for k, v in s.items():
        print(f"{k}: {v}")

    # Plot: show first 20 seconds to keep it readable
    tmax = 20.0
    idx = d.t <= tmax
    t = d.t[idx]
    imu = d.imu[idx, :]

    # Choose a couple of channels automatically (first 3) + magnitude
    n_show = min(3, imu.shape[1])
    plt.figure()
    for i in range(n_show):
        plt.plot(t, imu[:, i], label=d.imu_channels[i])
    plt.title(f"{args.id}: First IMU channels (first {tmax:.0f}s)")
    plt.xlabel("t [s]")
    plt.legend()

    # Magnitude plot (helps visually)
    plt.figure()
    mag = np.linalg.norm(imu, axis=1)
    plt.plot(t, mag)
    plt.title(f"{args.id}: IMU magnitude (first {tmax:.0f}s)")
    plt.xlabel("t [s]")
    plt.ylabel("||imu||")

    # Event markers
    plt.figure()
    plt.plot(t, mag)
    ev_t = np.array(d.events["time_s"])
    ev_t = ev_t[ev_t <= tmax]
    for et in ev_t:
        plt.axvline(et, linestyle="--", linewidth=1)
    plt.title(f"{args.id}: Events from .ts over IMU magnitude")
    plt.xlabel("t [s]")
    plt.ylabel("||imu||")

    plt.show()


if __name__ == "__main__":
    main()
