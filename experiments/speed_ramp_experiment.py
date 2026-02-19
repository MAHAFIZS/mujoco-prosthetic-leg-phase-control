# experiments/speed_ramp_experiment.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results/day3_runs")
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=42)

    # speed ramp definition (cyc/s)
    ap.add_argument("--omega0", type=float, default=0.7)
    ap.add_argument("--omega1", type=float, default=1.3)
    ap.add_argument("--t_ramp_start", type=float, default=4.0)
    ap.add_argument("--t_ramp_end", type=float, default=12.0)

    # channel corruptions
    ap.add_argument("--latency_ms", type=int, default=100)
    ap.add_argument("--jitter_flip_p", type=float, default=0.05)
    ap.add_argument("--conf_dropout_p", type=float, default=0.02)
    ap.add_argument("--conf_dropout_len_ms", type=int, default=300)
    ap.add_argument("--conf_min", type=float, default=0.20)

    # estimator methods
    ap.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=["kf_event"],
        choices=["integrate", "kf_noevent", "kf_event"],
        help="Run one or more methods. Example: --methods integrate kf_noevent kf_event",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    ramp_dir = out_root / "speed_ramp"
    ramp_dir.mkdir(parents=True, exist_ok=True)

    py = str(Path(".venv/Scripts/python.exe")) if sys.platform.startswith("win") else sys.executable

    for method in args.methods:
        # ---- SIM ----
        csv_path = ramp_dir / f"{method}.csv"
        sim_plot = ramp_dir / f"{method}.png"

        cmd_sim = [
            py, "-m", "experiments.run_leg_demo",
            "--seconds", str(args.seconds),
            "--use_stream_phase_gait",
            "--no_viewer",
            "--no_sleep",
            "--method", method,
            "--latency_ms", str(args.latency_ms),
            "--jitter_flip_p", str(args.jitter_flip_p),
            "--conf_dropout_p", str(args.conf_dropout_p),
            "--conf_dropout_len_ms", str(args.conf_dropout_len_ms),
            "--seed", str(args.seed),
            "--log_csv", str(csv_path),
            "--save_plots", str(sim_plot),

            # speed profile args
            "--omega_profile", "ramp",
            "--omega0", str(args.omega0),
            "--omega1", str(args.omega1),
            "--t_ramp_start", str(args.t_ramp_start),
            "--t_ramp_end", str(args.t_ramp_end),
        ]
        run(cmd_sim)

        # ---- ANALYZE ----
        m_path = ramp_dir / f"{method}_metrics.json"
        err_plot = ramp_dir / f"{method}_metrics.png"
        rmse_time_plot = ramp_dir / f"{method}_rmse_vs_time.png"

        cmd_an = [
            py, "experiments/analyze_run.py",
            "--csv", str(csv_path),
            "--out", str(m_path),
            "--plot", str(err_plot),
            "--conf_min", str(args.conf_min),

            # rolling RMSE plot
            "--rmse_time_plot", str(rmse_time_plot),
            "--rmse_win_s", "0.5",
            "--mark_ramp", str(args.t_ramp_start), str(args.t_ramp_end),
        ]
        run(cmd_an)

        print(f"\n[wrote] {m_path.resolve()}")
        print(f"[plot]  {rmse_time_plot.resolve()}")


if __name__ == "__main__":
    main()
