# experiments/day3_sweep.py
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _print_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))


def run(cmd: list[str]) -> None:
    _print_cmd(cmd)
    subprocess.run(cmd, check=True)


def load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def get_rmse(m: dict[str, Any], prefer_lagcomp: bool = True) -> float:
    """
    RMSE selection matching analyze_run.py schema:

      phase_rmse_deg_lagcomp
      phase_rmse_deg_raw

    with fallback to older keys if needed.
    """
    if prefer_lagcomp and "phase_rmse_deg_lagcomp" in m:
        return float(m["phase_rmse_deg_lagcomp"])
    if "phase_rmse_deg_raw" in m:
        return float(m["phase_rmse_deg_raw"])

    # Fallbacks (older versions)
    if "phase_rmse_deg" in m:
        return float(m["phase_rmse_deg"])
    if (
        "phase_error_deg" in m
        and isinstance(m["phase_error_deg"], dict)
        and "rmse_deg" in m["phase_error_deg"]
    ):
        return float(m["phase_error_deg"]["rmse_deg"])

    raise KeyError(f"RMSE key not found. Available keys: {sorted(m.keys())}")


def get_mae(m: dict[str, Any], prefer_lagcomp: bool = True) -> float:
    if prefer_lagcomp and "phase_mae_deg_lagcomp" in m:
        return float(m["phase_mae_deg_lagcomp"])
    if "phase_mae_deg_raw" in m:
        return float(m["phase_mae_deg_raw"])

    # Fallbacks
    if "phase_mae_deg" in m:
        return float(m["phase_mae_deg"])
    if (
        "phase_error_deg" in m
        and isinstance(m["phase_error_deg"], dict)
        and "mae_deg" in m["phase_error_deg"]
    ):
        return float(m["phase_error_deg"]["mae_deg"])

    raise KeyError(f"MAE key not found. Available keys: {sorted(m.keys())}")


def detect_python_exe() -> str:
    """
    Prefer current venv python if present; otherwise sys.executable.
    Works on Windows/Linux.
    """
    # If user is already in a venv, sys.executable is correct.
    py = sys.executable

    # If launched from repo root, prefer local .venv if exists.
    if sys.platform.startswith("win"):
        cand = Path(".venv/Scripts/python.exe")
    else:
        cand = Path(".venv/bin/python")

    if cand.exists():
        py = str(cand)

    return py


@dataclass(frozen=True)
class Job:
    cond: str
    method: str
    lat: int
    jit: float
    drop: float
    seconds: float
    seed: int
    conf_dropout_len_ms: int
    conf_min: float
    prefer_lagcomp: bool
    out_dir: Path
    py: str
    force: bool


def outputs_for(job: Job) -> dict[str, Path]:
    cond_dir = job.out_dir / job.cond
    return {
        "cond_dir": cond_dir,
        "csv": cond_dir / f"{job.method}.csv",
        "sim_plot": cond_dir / f"{job.method}.png",
        "metrics": cond_dir / f"{job.method}_metrics.json",
        "metrics_plot": cond_dir / f"{job.method}_metrics.png",
    }


def needs_run(job: Job) -> bool:
    out = outputs_for(job)
    if job.force:
        return True
    # Minimum artifacts we want
    return not (out["csv"].exists() and out["metrics"].exists() and out["metrics_plot"].exists())


def run_one(job: Job) -> Tuple[str, str]:
    """
    Run one (condition, method) and return (cond, method).
    Designed for ProcessPoolExecutor.
    """
    out = outputs_for(job)
    cond_dir = out["cond_dir"]
    cond_dir.mkdir(parents=True, exist_ok=True)

    if not needs_run(job):
        print(f"[skip] {job.cond} / {job.method} (already done)")
        return (job.cond, job.method)

    cmd_sim = [
        job.py, "-m", "experiments.run_leg_demo",
        "--seconds", str(job.seconds),
        "--use_stream_phase_gait",
        "--no_viewer",
        "--no_sleep",
        "--method", job.method,
        "--latency_ms", str(job.lat),
        "--jitter_flip_p", str(job.jit),
        "--conf_dropout_p", str(job.drop),
        "--conf_dropout_len_ms", str(job.conf_dropout_len_ms),
        "--seed", str(job.seed),
        "--log_csv", str(out["csv"]),
        "--save_plots", str(out["sim_plot"]),
    ]
    run(cmd_sim)

    cmd_an = [
        job.py, "experiments/analyze_run.py",
        "--csv", str(out["csv"]),
        "--out", str(out["metrics"]),
        "--plot", str(out["metrics_plot"]),
        "--conf_min", str(job.conf_min),
    ]
    run(cmd_an)

    return (job.cond, job.method)


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    # stable column ordering
    cols = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in cols:
                cols.append(k)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--latencies_ms", type=int, nargs="*", default=[0, 100, 200])
    ap.add_argument("--jitters", type=float, nargs="*", default=[0.0, 0.05])
    ap.add_argument("--dropouts", type=float, nargs="*", default=[0.0, 0.02])

    ap.add_argument("--conf_dropout_len_ms", type=int, default=300)
    ap.add_argument("--conf_min", type=float, default=0.20)

    ap.add_argument("--prefer_lagcomp", action="store_true", default=True)

    ap.add_argument("--out_dir", type=str, default="results/day3_runs")

    ap.add_argument("--force", action="store_true", help="rerun even if outputs exist")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (safe default: 1)")

    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    py = detect_python_exe()
    print(f"[python] {py}")

    methods = ["integrate", "kf_noevent", "kf_event"]  # OptionB = kf_event

    # --- Build jobs ---
    jobs: list[Job] = []
    for lat in args.latencies_ms:
        for jit in args.jitters:
            for drop in args.dropouts:
                cond = f"lat{lat}_jit{jit}_drop{drop}"
                for method in methods:
                    jobs.append(Job(
                        cond=cond,
                        method=method,
                        lat=lat,
                        jit=jit,
                        drop=drop,
                        seconds=args.seconds,
                        seed=args.seed,
                        conf_dropout_len_ms=args.conf_dropout_len_ms,
                        conf_min=args.conf_min,
                        prefer_lagcomp=bool(args.prefer_lagcomp),
                        out_dir=out_root,
                        py=py,
                        force=bool(args.force),
                    ))

    # --- Execute ---
    if args.jobs <= 1:
        for j in jobs:
            run_one(j)
    else:
        # Windows: ProcessPoolExecutor requires this file to be import-safe (it is).
        # Note: MuJoCo may be heavy; if it becomes unstable, set --jobs 1.
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(run_one, j) for j in jobs]
            for fut in as_completed(futs):
                fut.result()

    # --- Load metrics and build condition summary ---
    summary_rows: list[dict[str, Any]] = []

    for lat in args.latencies_ms:
        for jit in args.jitters:
            for drop in args.dropouts:
                cond = f"lat{lat}_jit{jit}_drop{drop}"
                cond_dir = out_root / cond

                metrics_by_method: Dict[str, Dict[str, Any]] = {}
                for method in methods:
                    m_path = cond_dir / f"{method}_metrics.json"
                    if not m_path.exists():
                        raise FileNotFoundError(f"Missing metrics: {m_path}")
                    metrics_by_method[method] = load_json(m_path)

                m_base = metrics_by_method["integrate"]
                m_noev = metrics_by_method["kf_noevent"]
                m_optb = metrics_by_method["kf_event"]

                rmse_base = get_rmse(m_base, prefer_lagcomp=args.prefer_lagcomp)
                rmse_noev = get_rmse(m_noev, prefer_lagcomp=args.prefer_lagcomp)
                rmse_optb = get_rmse(m_optb, prefer_lagcomp=args.prefer_lagcomp)

                mae_base = get_mae(m_base, prefer_lagcomp=args.prefer_lagcomp)
                mae_noev = get_mae(m_noev, prefer_lagcomp=args.prefer_lagcomp)
                mae_optb = get_mae(m_optb, prefer_lagcomp=args.prefer_lagcomp)

                if rmse_base > 1e-12:
                    improvement = 100.0 * (rmse_base - rmse_optb) / rmse_base
                else:
                    improvement = 0.0

                print(f"\n[COND {cond}] OptionB RMSE improvement vs integrate: {improvement:.1f}%")

                summary_rows.append({
                    "condition": cond,
                    "latency_ms": lat,
                    "jitter_flip_p": jit,
                    "conf_dropout_p": drop,
                    "prefer_lagcomp": bool(args.prefer_lagcomp),

                    "integrate_mae_deg": mae_base,
                    "integrate_rmse_deg": rmse_base,

                    "kf_noevent_mae_deg": mae_noev,
                    "kf_noevent_rmse_deg": rmse_noev,

                    "kf_event_mae_deg": mae_optb,
                    "kf_event_rmse_deg": rmse_optb,

                    "optionB_improvement_pct_vs_integrate": improvement,

                    # keep extra info if present
                    "kf_event_phase_lag_ms_est": float(m_optb.get("phase_lag_ms_est", 0.0)),
                    "kf_event_phase_lag_samples_est": int(m_optb.get("phase_lag_samples_est", 0)),
                    "kf_event_gated_fraction": float(m_optb.get("gated_fraction_conf_lt_conf_min", 0.0)),
                })

    out_summary_json = out_root / "summary.json"
    out_summary_csv = out_root / "summary.csv"
    out_summary_json.write_text(json.dumps(summary_rows, indent=2))
    write_summary_csv(out_summary_csv, summary_rows)

    print(f"\n[wrote] {out_summary_json.resolve()}")
    print(f"[wrote] {out_summary_csv.resolve()}")

    # Optional: auto-generate report if available
    report_py = Path("experiments/make_report.py")
    if report_py.exists():
        try:
            run([py, "experiments/make_report.py", "--in_dir", str(out_root)])
        except Exception as e:
            print(f"[warn] make_report failed: {e}")


if __name__ == "__main__":
    main()
