# experiments/run_leg_demo.py
# Dummy human + prosthetic leg in MuJoCo
# Streamed binary stance/swing phase -> continuous phase in [0,1)
# Day3-ready: logs phi_ref (clean) vs phi_obs (corrupted) vs phi_out (method)

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from src.realtime_phase import stream_proxy_phases_from_uci


# ----------------------------
# Utilities
# ----------------------------
def mj_id_or_neg(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    return mujoco.mj_name2id(model, obj, name)


def safe_mj_name2id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj, name)
    if idx < 0:
        raise ValueError(f"Name not found in model: {name} (obj={obj})")
    return idx


def clip_ctrl(model: mujoco.MjModel, aidx: np.ndarray, u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float).copy()
    lo = model.actuator_ctrlrange[aidx, 0]
    hi = model.actuator_ctrlrange[aidx, 1]
    return np.minimum(np.maximum(u, lo), hi)


def wrap01(x: float) -> float:
    x = x % 1.0
    return x if x >= 0 else x + 1.0


def wrap_pm05(x: float) -> float:
    """wrap to [-0.5, 0.5) in cycles"""
    return (x + 0.5) % 1.0 - 0.5


def omega_at_time(now_t: float, profile: str, omega0: float, omega1: float, t0: float, t1: float) -> float:
    if profile == "const":
        return float(omega0)
    # ramp
    if t1 <= t0:
        return float(omega1)
    if now_t <= t0:
        return float(omega0)
    if now_t >= t1:
        return float(omega1)
    a = (now_t - t0) / (t1 - t0)
    return float(omega0 + a * (omega1 - omega0))


# ----------------------------
# Gait reference
# ----------------------------
def gait_reference_from_phase(phi: float) -> np.ndarray:
    """
    Low-lift gait-like joint targets for [hip, knee, ankle].
    phi in [0,1)
    """
    phi = float(phi % 1.0)
    hip = 0.05 * np.sin(2 * np.pi * phi)
    knee = -0.22 - 0.10 * (np.sin(np.pi * phi) ** 2)
    ankle = 0.00 + 0.22 * np.sin(2 * np.pi * phi - 0.6)
    return np.array([hip, knee, ankle], dtype=float)


# ----------------------------
# Synthetic stance/swing event stream for ramp experiments
# ----------------------------
@dataclass
class PhaseSample:
    phase: int
    confidence: float


def stream_synth_phase_events(
    hz: float,
    seconds: float,
    omega_fn: Callable[[float], float],
    stance_ratio: float = 0.65,
    base_conf: float = 0.8,
    conf_noise: float = 0.05,
    seed: int = 0,
) -> Iterator[PhaseSample]:
    """
    Generates stance/swing events from a time-varying cadence omega_fn(t) [cycles/s].
    Uses a continuous internal phase phi in [0,1). Stance if phi < stance_ratio.
    """
    rng = np.random.RandomState(seed)
    dt_tick = 1.0 / float(hz)
    n = int(round(float(seconds) * float(hz)))

    phi = 0.0
    for k in range(n):
        t = k * dt_tick
        omega = float(max(omega_fn(t), 1e-6))
        phi = (phi + omega * dt_tick) % 1.0

        phase = 1 if phi < float(stance_ratio) else 0
        conf = float(np.clip(base_conf + conf_noise * rng.randn(), 0.0, 1.0))
        yield PhaseSample(phase=phase, confidence=conf)


# ----------------------------
# Kalman phase predictor (cycles domain)
# ----------------------------
class KalmanPhasePredictor01:
    """
    KF on x=[phi_unwrapped, omega] where phi is in cycles (1.0 == full cycle).
    Measurement is wrapped phi in [0,1).
    """

    def __init__(self, dt: float, q_phi=1e-4, q_omega=5e-4, r=2e-3, gate_sigma=6.0):
        self.dt = float(dt)
        self.F = np.array([[1.0, self.dt],
                           [0.0, 1.0]], dtype=float)
        self.H = np.array([[1.0, 0.0]], dtype=float)
        self.Q = np.array([[q_phi, 0.0],
                           [0.0, q_omega]], dtype=float)
        self.R = np.array([[r]], dtype=float)

        self.x = np.zeros((2, 1), dtype=float)  # [phi_unwrapped, omega]
        self.P = np.diag([0.1, 1.0]).astype(float)

        self.initialized = False
        self.last_z_unwrapped: float | None = None
        self.gate_sigma = float(gate_sigma)

    @staticmethod
    def nearest_unwrapped(prev_unwrapped: float, z_wrapped01: float) -> float:
        k = round(prev_unwrapped - z_wrapped01)
        return z_wrapped01 + k

    def reset(self, phi_wrapped01: float, omega: float = 1.0):
        phi0 = float(wrap01(phi_wrapped01))
        self.x[:] = [[phi0], [omega]]
        self.P[:] = np.diag([0.02, 0.5])
        self.initialized = True
        self.last_z_unwrapped = phi0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[1, 0] = float(np.clip(self.x[1, 0], 0.2, 4.0))  # cycles/s clamp

    def update(self, phi_wrapped01: float):
        z_wrapped = float(wrap01(phi_wrapped01))
        if not self.initialized:
            self.reset(z_wrapped, omega=1.0)
            return

        assert self.last_z_unwrapped is not None
        z_unwrapped = self.nearest_unwrapped(self.last_z_unwrapped, z_wrapped)
        self.last_z_unwrapped = z_unwrapped
        z = np.array([[z_unwrapped]], dtype=float)

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R

        sigma = float(np.sqrt(S[0, 0]))
        if sigma > 0 and abs(float(y[0, 0])) > self.gate_sigma * sigma:
            return

        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        self.x[1, 0] = float(np.clip(self.x[1, 0], 0.2, 4.0))

    def step(self, phi_wrapped01: float, do_update: bool = True):
        self.predict()
        if do_update:
            self.update(phi_wrapped01)

    def omega(self) -> float:
        return float(self.x[1, 0])

    def predict_ahead(self, latency_s: float) -> float:
        return wrap01(float(self.x[0, 0] + self.x[1, 0] * latency_s))


# ----------------------------
# Phase builder (event-integrator)
# ----------------------------
@dataclass
class PhaseState:
    seg_phase: int = 1                 # stance=1 swing=0
    last_accept_t: float = 0.0
    last_good_phi: float = 0.0
    phi: float = 0.0
    last_hs_t: float | None = None     # last heel-strike time


def step_event_integrator(
    st: PhaseState,
    now_t: float,
    dt: float,
    gait_hz: float,
    applied_phase: int,
    applied_conf: float,
    conf_min: float,
    min_trans_s: float,
    use_stream_events: bool,
) -> tuple[float, bool, bool]:
    """
    Returns (phi, stance, heel_strike)
    - integrates phi += gait_hz*dt
    - if heel_strike (swing->stance), reset phi=0
    - debounces transitions
    - confidence gate holds phi
    """
    heel_strike = False

    # debounced transitions (only if using stream events and conf ok)
    if use_stream_events and (applied_conf >= conf_min):
        if applied_phase != st.seg_phase and (now_t - st.last_accept_t) >= min_trans_s:
            prev = st.seg_phase
            st.seg_phase = int(applied_phase)
            st.last_accept_t = now_t
            heel_strike = (prev == 0 and st.seg_phase == 1)

    stance = (st.seg_phase == 1)

    # nominal integrate
    st.phi = wrap01(st.phi + float(gait_hz) * float(dt))

    # heel strike reset
    if use_stream_events and heel_strike:
        st.phi = 0.0

    # confidence gate
    if applied_conf < conf_min:
        st.phi = st.last_good_phi
    else:
        st.last_good_phi = st.phi

    return float(st.phi), stance, heel_strike


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default="sim/dummy_human_prosthetic.xml")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=0.002)

    # omega profile (drives reference cadence)
    ap.add_argument("--omega_profile", type=str, default="const", choices=["const", "ramp"])
    ap.add_argument("--omega0", type=float, default=1.0)
    ap.add_argument("--omega1", type=float, default=1.0)
    ap.add_argument("--t_ramp_start", type=float, default=0.0)
    ap.add_argument("--t_ramp_end", type=float, default=0.0)

    # phase stream
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--mode", type=str, default="seq", choices=["seq", "random"])
    ap.add_argument("--which", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--uci_seed", type=int, default=0)

    # corruptions
    ap.add_argument("--latency_ms", type=float, default=0.0)
    ap.add_argument("--jitter_flip_p", type=float, default=0.0)
    ap.add_argument("--conf_dropout_p", type=float, default=0.0)
    ap.add_argument("--conf_dropout_len_ms", type=float, default=300.0)

    # gait nominal
    ap.add_argument("--gait_hz", type=float, default=1.0)
    ap.add_argument("--stance_ratio", type=float, default=0.65)

    # harness height
    ap.add_argument("--pelvis_z", type=float, default=0.95)

    # use stream events
    ap.add_argument("--use_stream_phase_gait", action="store_true")
    ap.add_argument("--conf_min", type=float, default=0.20)
    ap.add_argument("--min_transition_s", type=float, default=0.25)

    # method
    ap.add_argument("--method", type=str, default="kf_event",
                    choices=["integrate", "kf_noevent", "kf_event"])

    # Kalman params
    ap.add_argument("--kalman_q_phi", type=float, default=1e-4)
    ap.add_argument("--kalman_q_omega", type=float, default=5e-4)
    ap.add_argument("--kalman_r", type=float, default=2e-3)
    ap.add_argument("--kalman_gate_sigma", type=float, default=6.0)

    # Option B params
    ap.add_argument("--event_alpha", type=float, default=0.35)
    ap.add_argument("--min_cycle_s", type=float, default=0.45)
    ap.add_argument("--max_cycle_s", type=float, default=2.5)

    # misc
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_csv", type=str, default="")
    ap.add_argument("--save_plots", type=str, default="")
    ap.add_argument("--no_viewer", action="store_true")
    ap.add_argument("--no_sleep", action="store_true")
    ap.add_argument("--live_plot", action="store_true")
    ap.add_argument("--debug_contacts", action="store_true")
    args = ap.parse_args()

    np.random.seed(int(args.seed))

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path.resolve()}")
    print("Loading XML:", xml_path.resolve())

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = float(args.dt)
    data = mujoco.MjData(model)

    # omega(t) profile (cyc/s)
    def omega_true(t: float) -> float:
        if str(args.omega_profile) == "const":
            # backwards compat: if omega0 untouched and gait_hz differs, use gait_hz
            if abs(float(args.omega0) - 1.0) < 1e-12 and abs(float(args.gait_hz) - 1.0) > 1e-12:
                return float(max(args.gait_hz, 1e-3))
            return float(max(args.omega0, 1e-3))

        return float(max(
            omega_at_time(
                now_t=float(t),
                profile="ramp",
                omega0=float(args.omega0),
                omega1=float(args.omega1),
                t0=float(args.t_ramp_start),
                t1=float(args.t_ramp_end),
            ),
            1e-3
        ))

    # actuators
    a_pelvis = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pelvis_height")
    if a_pelvis < 0:
        print("[WARN] Actuator 'pelvis_height' not found. Pelvis may fall unless pelvis is fixed.")

    a_hip = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_r_hip")
    a_knee = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_r_knee")
    a_ankle = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_r_ankle")
    aidx_leg = np.array([a_hip, a_knee, a_ankle], dtype=int)

    # joints for init pose
    j_hip = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip")
    j_knee = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_knee")
    j_ankle = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_ankle")
    jidx_leg = np.array(
        [model.jnt_qposadr[j_hip], model.jnt_qposadr[j_knee], model.jnt_qposadr[j_ankle]],
        dtype=int,
    )

    # pelvis tz joint
    j_pelvis_tz = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tz")
    qadr_pelvis_tz = int(model.jnt_qposadr[j_pelvis_tz]) if j_pelvis_tz >= 0 else None

    # stance pin elements
    pin_mocap_body = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_BODY, "pin_mocap")
    rpin_site = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_SITE, "r_pin")
    eq_pin = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_EQUALITY, "stance_pin")
    if pin_mocap_body < 0 or rpin_site < 0 or eq_pin < 0:
        raise ValueError("XML missing stance pin elements (pin_mocap / r_pin / stance_pin).")

    pin_mocap_id = int(model.body_mocapid[pin_mocap_body])
    if pin_mocap_id < 0:
        raise ValueError("Body 'pin_mocap' exists but is not mocap='true' in the XML.")

    # init state
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if qadr_pelvis_tz is not None:
        data.qpos[qadr_pelvis_tz] = float(args.pelvis_z)
    data.qpos[jidx_leg] = np.array([0.0, -0.22, 0.0], dtype=float)
    mujoco.mj_forward(model, data)

    # ----------------------------
    # Select event stream
    # - For ramp: MUST use synthetic stream driven by omega_true(t)
    # - For const: use original UCI proxy stream
    # ----------------------------
    if str(args.omega_profile) == "ramp":
        phase_gen = stream_synth_phase_events(
            hz=float(args.hz),
            seconds=float(args.seconds),
            omega_fn=omega_true,
            stance_ratio=float(args.stance_ratio),
            seed=int(args.seed),
        )
    else:
        phase_steps = max(1, int(round(float(args.seconds) * float(args.hz))))
        phase_gen = stream_proxy_phases_from_uci(
            hz=float(args.hz),
            n_steps=int(phase_steps),
            mode=str(args.mode),
            which=str(args.which),
            seed=int(args.uci_seed),
            sleep=False,
        )

    # latency buffer (on observed phase/conf)
    latency_steps = int(round((float(args.latency_ms) / 1000.0) / float(args.dt)))
    obs_phase_buf = deque([1] * max(1, latency_steps + 1), maxlen=max(1, latency_steps + 1))
    obs_conf_buf = deque([0.8] * max(1, latency_steps + 1), maxlen=max(1, latency_steps + 1))

    # dropout scheduler in stream ticks (NOT sim dt)
    dropout_left_ticks = 0
    dropout_len_ticks = int(round((float(args.conf_dropout_len_ms) / 1000.0) * float(args.hz)))
    dropout_len_ticks = max(1, dropout_len_ticks)

    def set_pin_active(active: bool) -> None:
        data.eq_active[eq_pin] = 1 if active else 0

    def move_pin_to_current_foot() -> None:
        p = data.site_xpos[rpin_site].copy()
        data.mocap_pos[pin_mocap_id] = p
        data.mocap_quat[pin_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # start pinned
    move_pin_to_current_foot()
    set_pin_active(True)
    mujoco.mj_forward(model, data)
    pin_is_active = True

    # viewer
    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        time.sleep(0.05)

    # phase tick schedule
    phase_tick_interval = max(1, int(round(1.0 / (float(args.hz) * float(args.dt)))))
    raw_phase = 1
    raw_conf = 0.8

    # states:
    conf_min = float(args.conf_min)
    min_trans_s = max(float(args.min_transition_s), 0.0)
    use_stream_events = bool(args.use_stream_phase_gait)

    ref_state = PhaseState()
    obs_state = PhaseState()

    # kalman (runs on observed phi)
    latency_s = float(args.latency_ms) / 1000.0
    kf = KalmanPhasePredictor01(
        dt=float(args.dt),
        q_phi=float(args.kalman_q_phi),
        q_omega=float(args.kalman_q_omega),
        r=float(args.kalman_r),
        gate_sigma=float(args.kalman_gate_sigma),
    )
    kf.reset(phi_wrapped01=0.0, omega=float(max(omega_true(0.0), 1e-3)))

    # live plot
    live_plot = bool(args.live_plot)
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        PLOT_WINDOW_SEC = 8.0

        t_buf = []
        phi_ref_buf, phi_obs_buf, phi_out_buf = [], [], []
        conf_buf, omega_buf, omega_ref_buf = [], [], []

        (l_ref,) = ax[0].plot([], [], label="phi_ref (clean)")
        (l_obs,) = ax[0].plot([], [], "--", label="phi_obs (corrupted)")
        (l_out,) = ax[0].plot([], [], ":", label="phi_out (method)")
        ax[0].set_ylim(-0.05, 1.05)
        ax[0].legend(loc="upper right")
        ax[0].set_ylabel("phase (cycles)")

        (l_conf,) = ax[1].plot([], [], label="conf_obs")
        (l_omega,) = ax[1].plot([], [], "--", label="omega_out (cyc/s)")
        (l_oref,) = ax[1].plot([], [], ":", label="omega_ref (cyc/s)")
        ax[1].set_ylim(0.0, 3.5)
        ax[1].legend(loc="upper right")
        ax[1].set_ylabel("conf / omega")
        ax[1].set_xlabel("time [s]")

        PLOT_HZ = 20.0
        plot_every = max(1, int(round(1.0 / (PLOT_HZ * float(args.dt)))))

        def trim(now_t: float):
            while t_buf and (now_t - t_buf[0]) > PLOT_WINDOW_SEC:
                t_buf.pop(0)
                phi_ref_buf.pop(0); phi_obs_buf.pop(0); phi_out_buf.pop(0)
                conf_buf.pop(0); omega_buf.pop(0); omega_ref_buf.pop(0)
    else:
        plot_every = 10**9

    # logging
    log_path = Path(args.log_csv) if args.log_csv else None
    log_f = None
    log_writer = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", newline="")
        log_writer = csv.DictWriter(log_f, fieldnames=[
            "t", "dt",
            "raw_phase", "raw_conf",
            "obs_phase", "obs_conf",
            "stance_ref", "stance_obs",
            "heel_strike_ref", "heel_strike_obs",
            "phi_ref", "phi_obs", "phi_out",
            "omega_ref", "omega_out",
            "latency_ms", "jitter_flip_p", "conf_dropout_p",
            "method",
        ])
        log_writer.writeheader()

    # timing
    n_steps = int(round(float(args.seconds) / float(args.dt)))
    use_realtime = (not args.no_sleep)
    t0_wall = time.perf_counter()
    debug_every = max(1, int(round(0.2 / float(args.dt))))

    for step in range(n_steps):
        now_t = step * float(args.dt)
        gait_hz_now = float(max(omega_true(now_t), 1e-3))

        # update raw stream at hz
        if step % phase_tick_interval == 0:
            try:
                st = next(phase_gen)
                raw_phase = int(st.phase)
                raw_conf = float(getattr(st, "confidence", 0.5))
            except StopIteration:
                pass

            # apply jitter flip on OBSERVED signal
            obs_phase = raw_phase
            if float(args.jitter_flip_p) > 0 and (np.random.rand() < float(args.jitter_flip_p)):
                obs_phase = 1 - obs_phase

            # schedule confidence dropout on OBSERVED signal
            if dropout_left_ticks <= 0 and float(args.conf_dropout_p) > 0 and (np.random.rand() < float(args.conf_dropout_p)):
                dropout_left_ticks = dropout_len_ticks

            obs_conf = raw_conf
            if dropout_left_ticks > 0:
                obs_conf = 0.0
                dropout_left_ticks -= 1

            # push into latency buffers
            obs_phase_buf.append(int(obs_phase))
            obs_conf_buf.append(float(obs_conf))

        # get latency-applied observed values
        obs_phase_lat = int(obs_phase_buf[0]) if latency_steps > 0 else int(raw_phase)
        obs_conf_lat = float(obs_conf_buf[0]) if latency_steps > 0 else float(raw_conf)

        # ---- build reference (clean) phase from RAW stream ----
        phi_ref, stance_ref, hs_ref = step_event_integrator(
            ref_state, now_t, float(args.dt), gait_hz_now,
            applied_phase=int(raw_phase),
            applied_conf=float(raw_conf),
            conf_min=conf_min,
            min_trans_s=min_trans_s,
            use_stream_events=use_stream_events,
        )

        # ---- build observed phase from CORRUPTED stream ----
        phi_obs, stance_obs, hs_obs = step_event_integrator(
            obs_state, now_t, float(args.dt), gait_hz_now,
            applied_phase=int(obs_phase_lat),
            applied_conf=float(obs_conf_lat),
            conf_min=conf_min,
            min_trans_s=min_trans_s,
            use_stream_events=use_stream_events,
        )

        # stance pin based on observed stance (so sim stays stable)
        if stance_obs and (not pin_is_active):
            move_pin_to_current_foot()
            set_pin_active(True)
            pin_is_active = True
        elif (not stance_obs) and pin_is_active:
            set_pin_active(False)
            pin_is_active = False

        # ---- methods ----
        omega_out = gait_hz_now
        phi_out = phi_obs

        if args.method == "integrate":
            phi_out = phi_obs
            omega_out = gait_hz_now

        elif args.method == "kf_noevent":
            do_update = (obs_conf_lat >= conf_min)
            kf.step(phi_obs, do_update=do_update)
            phi_out = kf.predict_ahead(latency_s)
            omega_out = kf.omega()

        elif args.method == "kf_event":
            if hs_obs:
                # align phase boundary
                kf.x[0, 0] = round(kf.x[0, 0])

                # cadence correction from event timing
                if obs_state.last_hs_t is not None:
                    cycle_s = now_t - obs_state.last_hs_t
                    if float(args.min_cycle_s) <= cycle_s <= float(args.max_cycle_s):
                        omega_meas = 1.0 / cycle_s
                        alpha = float(args.event_alpha)
                        kf.x[1, 0] = (1.0 - alpha) * kf.x[1, 0] + alpha * omega_meas
                        kf.x[1, 0] = float(np.clip(kf.x[1, 0], 0.2, 4.0))
                obs_state.last_hs_t = now_t

            do_update = (obs_conf_lat >= conf_min) and (not hs_obs)
            kf.step(phi_obs, do_update=do_update)
            phi_out = kf.predict_ahead(latency_s)
            omega_out = kf.omega()

        # ---- control ----
        qref_leg = gait_reference_from_phase(phi_out)

        data.ctrl[:] = 0.0
        if a_pelvis >= 0:
            data.ctrl[a_pelvis] = float(args.pelvis_z)
        data.ctrl[aidx_leg] = clip_ctrl(model, aidx_leg, qref_leg)

        mujoco.mj_step(model, data)

        # debug
        if args.debug_contacts and (step % debug_every == 0):
            pz = data.qpos[qadr_pelvis_tz] if qadr_pelvis_tz is not None else float("nan")
            e_deg = wrap_pm05(phi_out - phi_ref) * 360.0
            print(
                f"t={now_t:5.2f}s raw={raw_phase} obs={obs_phase_lat} conf={obs_conf_lat:.2f} "
                f"phi_ref={phi_ref:.3f} phi_out={phi_out:.3f} e={e_deg:+6.2f}deg "
                f"omega_ref={gait_hz_now:.2f} omega_out={omega_out:.2f} "
                f"pin={'ON' if pin_is_active else 'OFF'} pelvis_z={pz:.3f}"
            )

        # sync viewer
        if viewer is not None:
            viewer.sync()

        # plot
        if live_plot:
            t_buf.append(now_t)
            phi_ref_buf.append(phi_ref)
            phi_obs_buf.append(phi_obs)
            phi_out_buf.append(phi_out)
            conf_buf.append(float(obs_conf_lat))
            omega_buf.append(float(omega_out))
            omega_ref_buf.append(float(gait_hz_now))
            trim(now_t)

            if step % plot_every == 0:
                l_ref.set_data(t_buf, phi_ref_buf)
                l_obs.set_data(t_buf, phi_obs_buf)
                l_out.set_data(t_buf, phi_out_buf)
                l_conf.set_data(t_buf, conf_buf)
                l_omega.set_data(t_buf, omega_buf)
                l_oref.set_data(t_buf, omega_ref_buf)
                ax[0].set_xlim(max(0.0, now_t - 8.0), now_t + 1e-3)
                fig.canvas.draw()
                fig.canvas.flush_events()

        # log
        if log_writer is not None:
            log_writer.writerow({
                "t": now_t,
                "dt": float(args.dt),
                "raw_phase": raw_phase,
                "raw_conf": float(raw_conf),
                "obs_phase": obs_phase_lat,
                "obs_conf": float(obs_conf_lat),
                "stance_ref": int(stance_ref),
                "stance_obs": int(stance_obs),
                "heel_strike_ref": int(hs_ref),
                "heel_strike_obs": int(hs_obs),
                "phi_ref": phi_ref,
                "phi_obs": phi_obs,
                "phi_out": phi_out,
                "omega_ref": float(gait_hz_now),
                "omega_out": float(omega_out),
                "latency_ms": float(args.latency_ms),
                "jitter_flip_p": float(args.jitter_flip_p),
                "conf_dropout_p": float(args.conf_dropout_p),
                "method": args.method,
            })

        # realtime pacing
        if use_realtime:
            target_wall = t0_wall + now_t
            while True:
                rem = target_wall - time.perf_counter()
                if rem <= 0:
                    break
                time.sleep(min(0.001, rem))

    if log_f is not None:
        log_f.close()
        print(f"[log] wrote: {log_path.resolve()}")

    if viewer is not None:
        viewer.close()
    if live_plot:
        plt.ioff()
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
