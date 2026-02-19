# experiments/analyze_run.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def wrap01(x: np.ndarray) -> np.ndarray:
    return np.mod(x, 1.0)


def wrap_pm05(x: np.ndarray) -> np.ndarray:
    """wrap to [-0.5, 0.5) cycles"""
    return (x + 0.5) % 1.0 - 0.5


def circ_err_deg(phi_pred: np.ndarray, phi_ref: np.ndarray) -> np.ndarray:
    """circular error in degrees in [-180,180)"""
    e_cyc = wrap_pm05(phi_pred - phi_ref)
    return e_cyc * 360.0


def rmse_deg(err_deg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err_deg))))


def mae_deg(err_deg: np.ndarray) -> float:
    return float(np.mean(np.abs(err_deg)))


@dataclass
class LagEstimate:
    lag_s: float
    lag_samples: int
    rmse_deg: float


def estimate_lag_by_scan(
    phi_pred: np.ndarray,
    phi_ref: np.ndarray,
    dt: float,
    max_lag_s: float = 0.30,
    step_s: float = 0.002,
) -> LagEstimate:
    """
    Robust lag estimation for circular phase signals.
    We scan candidate lags in [-max_lag_s, +max_lag_s] and pick the one minimizing RMSE.

    Convention:
      positive lag means: pred is delayed relative to ref -> shift pred forward (negative index) to align.
    """
    n = len(phi_ref)
    if n < 10:
        return LagEstimate(lag_s=0.0, lag_samples=0, rmse_deg=rmse_deg(circ_err_deg(phi_pred, phi_ref)))

    max_k = int(round(max_lag_s / dt))
    step_k = max(1, int(round(step_s / dt)))

    best = None
    for k in range(-max_k, max_k + 1, step_k):
        # Align arrays with shift k
        if k > 0:
            # pred delayed -> compare pred[k:] with ref[:-k]
            pred_al = phi_pred[k:]
            ref_al = phi_ref[:-k]
        elif k < 0:
            kk = -k
            pred_al = phi_pred[:-kk]
            ref_al = phi_ref[kk:]
        else:
            pred_al = phi_pred
            ref_al = phi_ref

        if len(pred_al) < 50:
            continue

        e = circ_err_deg(pred_al, ref_al)
        r = rmse_deg(e)
        if best is None or r < best.rmse_deg:
            best = LagEstimate(lag_s=k * dt, lag_samples=k, rmse_deg=r)

    if best is None:
        best = LagEstimate(lag_s=0.0, lag_samples=0, rmse_deg=rmse_deg(circ_err_deg(phi_pred, phi_ref)))
    return best


def rolling_rmse(
    phi_pred: np.ndarray,
    phi_ref: np.ndarray,
    dt: float,
    win_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """returns (t_center, rmse_deg_series)"""
    n = len(phi_ref)
    w = max(3, int(round(win_s / dt)))
    w = min(w, n)

    rmses = []
    t_cent = []
    for i in range(w, n + 1):
        a = i - w
        b = i
        e = circ_err_deg(phi_pred[a:b], phi_ref[a:b])
        rmses.append(rmse_deg(e))
        t_cent.append((a + b - 1) * 0.5 * dt)
    return np.array(t_cent), np.array(rmses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--plot", type=str, default="")
    ap.add_argument("--conf_min", type=float, default=0.20)

    # NEW: rolling RMSE vs time plot
    ap.add_argument("--rmse_time_plot", type=str, default="")
    ap.add_argument("--rmse_win_s", type=float, default=0.5)
    ap.add_argument("--mark_ramp", type=float, nargs=2, default=None)

    # NEW: lag scan params (robust)
    ap.add_argument("--lag_scan_max_s", type=float, default=0.30)
    ap.add_argument("--lag_scan_step_s", type=float, default=0.004)

    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load CSV
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Required columns
    for c in ["t", "dt", "phi_ref", "phi_out", "obs_conf", "omega_out"]:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {csv_path}. Found: {list(df.columns)}")

    t = df["t"].to_numpy(dtype=float)
    dt = float(np.median(df["dt"].to_numpy(dtype=float)))
    phi_ref = wrap01(df["phi_ref"].to_numpy(dtype=float))
    phi_out = wrap01(df["phi_out"].to_numpy(dtype=float))
    conf = df["obs_conf"].to_numpy(dtype=float)
    omega_out = df["omega_out"].to_numpy(dtype=float)

    # Gate by confidence
    keep = conf >= float(args.conf_min)
    gated_fraction = 1.0 - float(np.mean(keep))

    # If everything is gated, fallback to un-gated stats (but still report gated fraction)
    if np.sum(keep) < 50:
        keep = np.ones_like(keep, dtype=bool)

    phi_ref_k = phi_ref[keep]
    phi_out_k = phi_out[keep]

    # RAW error
    e_raw = circ_err_deg(phi_out_k, phi_ref_k)
    mae_raw = mae_deg(e_raw)
    rmse_raw = rmse_deg(e_raw)

    # Robust lag estimate (scan)
    lag = estimate_lag_by_scan(
        phi_pred=phi_out_k,
        phi_ref=phi_ref_k,
        dt=dt,
        max_lag_s=float(args.lag_scan_max_s),
        step_s=float(args.lag_scan_step_s),
    )

    # Lag-compensated error
    k = lag.lag_samples
    if k > 0:
        phi_out_lc = phi_out_k[k:]
        phi_ref_lc = phi_ref_k[:-k]
    elif k < 0:
        kk = -k
        phi_out_lc = phi_out_k[:-kk]
        phi_ref_lc = phi_ref_k[kk:]
    else:
        phi_out_lc = phi_out_k
        phi_ref_lc = phi_ref_k

    e_lc = circ_err_deg(phi_out_lc, phi_ref_lc)
    mae_lc = mae_deg(e_lc)
    rmse_lc = rmse_deg(e_lc)

    # Omega stats
    omega_mean = float(np.mean(omega_out))
    omega_std = float(np.std(omega_out))

    metrics: dict[str, Any] = {
        "samples": int(len(df)),
        "dt_s": float(dt),

        "phase_mae_deg_raw": float(mae_raw),
        "phase_rmse_deg_raw": float(rmse_raw),

        "phase_lag_s_est": float(lag.lag_s),
        "phase_lag_ms_est": float(lag.lag_s * 1000.0),
        "phase_lag_samples_est": int(lag.lag_samples),

        "phase_mae_deg_lagcomp": float(mae_lc),
        "phase_rmse_deg_lagcomp": float(rmse_lc),

        "omega_mean_cyc_s": float(omega_mean),
        "omega_std_cyc_s": float(omega_std),

        "conf_min": float(args.conf_min),
        "gated_fraction_conf_lt_conf_min": float(gated_fraction),
    }

    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[metrics] wrote: {out_path.resolve()}")

    # Plot: error histogram / time series summary (optional)
    if args.plot:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 4))
        plt.hist(e_raw, bins=60)
        plt.xlabel("phase error [deg] (raw)")
        plt.ylabel("count")
        plt.title(f"Error hist | raw RMSE={rmse_raw:.1f}° | lagcomp RMSE={rmse_lc:.1f}° | lag={lag.lag_s*1000:.0f}ms")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
        print(f"[plot] saved: {plot_path.resolve()}")

    # Rolling RMSE vs time (optional)
    if args.rmse_time_plot:
        rmse_plot_path = Path(args.rmse_time_plot)
        rmse_plot_path.parent.mkdir(parents=True, exist_ok=True)

        # Use full arrays but mask conf by replacing low-conf with NaN and interpolate short gaps
        phi_ref_f = phi_ref.copy()
        phi_out_f = phi_out.copy()
        bad = conf < float(args.conf_min)
        phi_ref_f[bad] = np.nan
        phi_out_f[bad] = np.nan

        # simple forward-fill then back-fill (good enough for rolling windows)
        def _fill(a: np.ndarray) -> np.ndarray:
            a2 = a.copy()
            # forward fill
            last = np.nan
            for i in range(len(a2)):
                if np.isnan(a2[i]):
                    a2[i] = last
                else:
                    last = a2[i]
            # back fill
            last = np.nan
            for i in range(len(a2)-1, -1, -1):
                if np.isnan(a2[i]):
                    a2[i] = last
                else:
                    last = a2[i]
            # any remaining nan -> 0
            a2 = np.nan_to_num(a2, nan=0.0)
            return a2

        phi_ref_f = wrap01(_fill(phi_ref_f))
        phi_out_f = wrap01(_fill(phi_out_f))

        # Apply lag compensation to rolling series too:
        k_full = lag.lag_samples
        if k_full > 0:
            phi_out_roll = phi_out_f[k_full:]
            phi_ref_roll = phi_ref_f[:-k_full]
            t_roll = t[:-k_full]
        elif k_full < 0:
            kk = -k_full
            phi_out_roll = phi_out_f[:-kk]
            phi_ref_roll = phi_ref_f[kk:]
            t_roll = t[kk:]
        else:
            phi_out_roll = phi_out_f
            phi_ref_roll = phi_ref_f
            t_roll = t

        tt, rr = rolling_rmse(phi_out_roll, phi_ref_roll, dt=dt, win_s=float(args.rmse_win_s))

        plt.figure(figsize=(10, 4))
        plt.plot(tt, rr)
        plt.xlabel("time [s]")
        plt.ylabel(f"rolling RMSE [deg] (win={args.rmse_win_s:.2f}s)")
        title = f"RMSE vs time (lag-comp) | lag={lag.lag_s*1000:.0f}ms"
        plt.title(title)

        if args.mark_ramp is not None:
            t0, t1 = float(args.mark_ramp[0]), float(args.mark_ramp[1])
            plt.axvline(t0, linestyle="--")
            plt.axvline(t1, linestyle="--")

        plt.tight_layout()
        plt.savefig(rmse_plot_path, dpi=160)
        plt.close()
        print(f"[plot] saved: {rmse_plot_path.resolve()}")

    # Console summary
    print("\n=== Summary ===")
    print(f"Samples: {len(df)}  dt≈{dt:.6f}s")
    print(f"Phase MAE RAW: {mae_raw:.3f} deg (RMSE {rmse_raw:.3f} deg)")
    print(f"Phase lag (est): {metrics['phase_lag_ms_est']:.1f} ms  (samples {metrics['phase_lag_samples_est']})")
    print(f"Phase MAE LAG-COMP: {mae_lc:.3f} deg (RMSE {rmse_lc:.3f} deg)")
    print(f"Omega: mean {omega_mean:.3f} cyc/s  std {omega_std:.3f}")
    print(f"Gated fraction (conf<{args.conf_min:.2f}): {gated_fraction:.3f}")
    print(f"\n[wrote] {out_path.resolve()}")


if __name__ == "__main__":
    main()
