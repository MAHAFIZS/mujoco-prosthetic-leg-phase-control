# experiments/make_report.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def pick_metric(m: dict[str, Any], key: str, fallback: list[str] | None = None) -> float:
    if key in m:
        return safe_float(m[key])
    if fallback:
        for k in fallback:
            if k in m:
                return safe_float(m[k])
    return float("nan")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_\n"
    return df.to_markdown(index=False) + "\n"


# ------------------------
# Day3 plots/tables
# ------------------------
def plot_rmse_vs_latency(summary_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(summary_csv)

    plt.figure(figsize=(8, 4))
    for method, col in [
        ("integrate", "integrate_rmse_deg"),
        ("kf_noevent", "kf_noevent_rmse_deg"),
        ("kf_event", "kf_event_rmse_deg"),
    ]:
        if col not in df.columns:
            continue
        g = df.groupby("latency_ms")[col].mean().reset_index()
        plt.plot(g["latency_ms"], g[col], marker="o", label=method)

    plt.xlabel("latency [ms]")
    plt.ylabel("RMSE [deg]")
    plt.title("RMSE vs latency (mean across jitter/dropout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_improvement_sorted(summary_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(summary_csv)
    col = "optionB_improvement_pct_vs_integrate"
    if col not in df.columns:
        return

    d = df.sort_values(col).reset_index(drop=True)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(d)), d[col].to_numpy(), marker="o")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("condition (sorted)")
    plt.ylabel("improvement [%] (kf_event vs integrate)")
    plt.title("OptionB improvement sorted across conditions")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def write_robustness_table(summary_csv: Path, out_md: Path) -> None:
    df = pd.read_csv(summary_csv)

    rows = []
    for lat in sorted(df["latency_ms"].unique()):
        dlat = df[df["latency_ms"] == lat]
        rows.append({
            "latency_ms": int(lat),
            "integrate_rmse_mean": float(dlat["integrate_rmse_deg"].mean()),
            "kf_noevent_rmse_mean": float(dlat["kf_noevent_rmse_deg"].mean()),
            "kf_event_rmse_mean": float(dlat["kf_event_rmse_deg"].mean()),
            "kf_event_impr_%_mean": float(dlat["optionB_improvement_pct_vs_integrate"].mean()),
        })

    out_df = pd.DataFrame(rows).round(3)
    out_md.write_text("# Robustness summary (Day3 sweep)\n\n" + md_table(out_df), encoding="utf-8")


# ------------------------
# Speed ramp
# ------------------------
def load_speed_ramp_metrics(speed_ramp_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method in ["integrate", "kf_noevent", "kf_event"]:
        mp = speed_ramp_dir / f"{method}_metrics.json"
        if not mp.exists():
            continue
        m = load_json(mp)

        rows.append({
            "method": method,
            "rmse_deg_raw": pick_metric(m, "phase_rmse_deg_raw", ["phase_rmse_deg"]),
            "mae_deg_raw": pick_metric(m, "phase_mae_deg_raw", ["phase_mae_deg"]),
            "lag_ms_est": pick_metric(m, "phase_lag_ms_est", []),
            "rmse_deg_lagcomp": pick_metric(m, "phase_rmse_deg_lagcomp", []),
            "mae_deg_lagcomp": pick_metric(m, "phase_mae_deg_lagcomp", []),
            "gated_fraction": pick_metric(m, "gated_fraction_conf_lt_conf_min", []),
            "omega_mean": pick_metric(m, "omega_mean_cyc_s", ["omega_mean"]),
            "omega_std": pick_metric(m, "omega_std_cyc_s", ["omega_std"]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # compute improvement vs integrate (prefer lagcomp if present)
    base = df[df["method"] == "integrate"]
    if len(base) == 1:
        base_rmse = float(base["rmse_deg_lagcomp"].iloc[0])
        if not np.isfinite(base_rmse):
            base_rmse = float(base["rmse_deg_raw"].iloc[0])

        impr = []
        for _, r in df.iterrows():
            rmse = float(r["rmse_deg_lagcomp"])
            if not np.isfinite(rmse):
                rmse = float(r["rmse_deg_raw"])
            if base_rmse > 1e-12 and np.isfinite(rmse):
                impr.append(100.0 * (base_rmse - rmse) / base_rmse)
            else:
                impr.append(np.nan)
        df["improvement_%_vs_integrate"] = impr

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    report_dir = in_dir / "report"
    ensure_dir(report_dir)

    summary_csv = in_dir / "summary.csv"

    # outputs
    rmse_vs_latency_png = report_dir / "rmse_vs_latency.png"
    improvement_sorted_png = report_dir / "improvement_sorted.png"
    robustness_md = report_dir / "robustness_table.md"
    speed_ramp_md = report_dir / "speed_ramp_table.md"
    summary_md = report_dir / "summary.md"

    # -------- Day3 sweep section --------
    if summary_csv.exists():
        plot_rmse_vs_latency(summary_csv, rmse_vs_latency_png)
        print(f"[plot] {rmse_vs_latency_png.resolve()}")

        plot_improvement_sorted(summary_csv, improvement_sorted_png)
        print(f"[plot] {improvement_sorted_png.resolve()}")

        write_robustness_table(summary_csv, robustness_md)
        print(f"[wrote] {robustness_md.resolve()}")
    else:
        print(f"[warn] missing {summary_csv.resolve()}")

    # -------- Speed ramp section --------
    speed_ramp_dir = in_dir / "speed_ramp"
    ramp_df = load_speed_ramp_metrics(speed_ramp_dir)

    if ramp_df.empty:
        print(f"[info] no speed ramp metrics found in {speed_ramp_dir.resolve()}")
    else:
        speed_ramp_md.write_text(
            "# Speed ramp robustness\n\n" + md_table(ramp_df.round(3)),
            encoding="utf-8",
        )
        print(f"[wrote] {speed_ramp_md.resolve()}")

    # -------- summary.md --------
    parts: List[str] = []
    parts.append("# Phase robustness report\n")

    if rmse_vs_latency_png.exists():
        parts.append("## Day3 sweep: RMSE vs latency\n")
        parts.append(f"![rmse_vs_latency]({rmse_vs_latency_png.name})\n")

    if improvement_sorted_png.exists():
        parts.append("## Day3 sweep: OptionB improvement sorted\n")
        parts.append(f"![improvement_sorted]({improvement_sorted_png.name})\n")

    if robustness_md.exists():
        parts.append("## Day3 sweep: robustness table\n")
        parts.append(robustness_md.read_text(encoding="utf-8") + "\n")

    parts.append("## Speed ramp experiment\n")
    if not ramp_df.empty:
        parts.append(speed_ramp_md.read_text(encoding="utf-8") + "\n")
        parts.append("### RMSE vs time plots\n")
        for method in ["integrate", "kf_noevent", "kf_event"]:
            p = speed_ramp_dir / f"{method}_rmse_vs_time.png"
            if p.exists():
                # path relative from report/ -> ../speed_ramp/...
                parts.append(f"**{method}**\n\n![{method}_rmse_vs_time](../speed_ramp/{p.name})\n")
            else:
                parts.append(f"- missing: {p}\n")
    else:
        parts.append("_No speed ramp results found._\n")

    summary_md.write_text("\n".join(parts), encoding="utf-8")
    print(f"[wrote] {summary_md.resolve()}")


if __name__ == "__main__":
    main()
