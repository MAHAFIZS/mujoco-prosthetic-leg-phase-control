from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.io_dataset import load_record
from src.labels import label_record_two_class

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=str, default="als1")
    args = ap.parse_args()

    d = load_record(args.id)
    lab = label_record_two_class(d.t, d.events)

    print("Label info:", lab.events_used)

    # Plot first 25s
    tmax = 25.0
    idx = d.t <= tmax

    imu = d.imu[idx]
    t = d.t[idx]

    mag = np.linalg.norm(imu, axis=1)

    plt.figure()
    plt.plot(t, mag)
    plt.step(t, lab.contact[idx] * mag.max(), where="post")  # overlay contact
    plt.title(f"{args.id}: IMU magnitude + inferred contact (first {tmax:.0f}s)")
    plt.xlabel("t [s]")
    plt.ylabel("||imu|| (contact scaled)")

    plt.show()

if __name__ == "__main__":
    main()
