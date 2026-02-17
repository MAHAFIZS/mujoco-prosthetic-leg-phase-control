from __future__ import annotations
import numpy as np

from src.io_dataset import load_record
from src.labels import label_record_two_class
from src.windowing import window_signal
from src.window_labels import window_labels_binary

def main():
    d = load_record("als1")
    lab = label_record_two_class(d.t, d.events)

    fs = int(round(1 / np.mean(np.diff(d.t))))
    print("Sampling rate:", fs)

    win_len = int(0.2 * fs)  # 200 ms
    step = int(0.05 * fs)    # 50 ms

    X = window_signal(d.imu, win_len, step)
    y = window_labels_binary(lab.contact, win_len, step)

    print("Window shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Stance fraction (window-level):", y.mean())

if __name__ == "__main__":
    main()
