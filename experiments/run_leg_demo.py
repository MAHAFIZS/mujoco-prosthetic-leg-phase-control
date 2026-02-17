from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from src.realtime_phase import stream_proxy_phases_from_uci


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, default="sim/dummy_human_prosthetic.xml")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=0.002)

    # phase stream
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--mode", type=str, default="seq", choices=["seq", "random"])
    ap.add_argument("--which", type=str, default="train", choices=["train", "test"])
    ap.add_argument("--uci_seed", type=int, default=0)
    ap.add_argument("--latency_ms", type=float, default=0.0)

    # gait timing (used for mapping binary phase -> continuous phi)
    ap.add_argument("--gait_hz", type=float, default=1.0)
    ap.add_argument("--stance_ratio", type=float, default=0.65)

    # harness height
    ap.add_argument("--pelvis_z", type=float, default=0.95)

    # NEW: drive gait from streamed phase (recommended)
    ap.add_argument("--use_stream_phase_gait", action="store_true",
                    help="Drive continuous gait phase phi from streamed binary phase.")
    ap.add_argument("--conf_min", type=float, default=0.20,
                    help="If confidence < conf_min, hold phi (prevents jitter).")

    # runtime
    ap.add_argument("--no_viewer", action="store_true")
    ap.add_argument("--no_sleep", action="store_true")
    ap.add_argument("--live_plot", action="store_true")
    ap.add_argument("--debug_contacts", action="store_true")

    args = ap.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path.resolve()}")
    print("Loading XML:", xml_path.resolve())

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = float(args.dt)
    data = mujoco.MjData(model)

    # actuators (pelvis actuator is optional; warn if missing)
    a_pelvis = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pelvis_height")
    if a_pelvis < 0:
        print("[WARN] Actuator 'pelvis_height' not found. Pelvis may fall unless pelvis is fixed.")

    a_hip = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_r_hip")
    a_knee = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_r_knee")
    a_ankle = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_r_ankle")
    aidx_leg = np.array([a_hip, a_knee, a_ankle], dtype=int)

    # leg joints for init pose
    j_hip = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_hip")
    j_knee = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_knee")
    j_ankle = safe_mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "r_ankle")
    jidx_leg = np.array(
        [model.jnt_qposadr[j_hip], model.jnt_qposadr[j_knee], model.jnt_qposadr[j_ankle]],
        dtype=int,
    )

    # pelvis tz joint (optional; for initial height logging)
    j_pelvis_tz = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tz")
    qadr_pelvis_tz = int(model.jnt_qposadr[j_pelvis_tz]) if j_pelvis_tz >= 0 else None

    # stance pin elements
    pin_mocap_body = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_BODY, "pin_mocap")
    rpin_site = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_SITE, "r_pin")
    eq_pin = mj_id_or_neg(model, mujoco.mjtObj.mjOBJ_EQUALITY, "stance_pin")

    missing = []
    if pin_mocap_body < 0:
        missing.append("body 'pin_mocap' (mocap=true) in <worldbody>")
    if rpin_site < 0:
        missing.append("site 'r_pin' inside right_ankle body")
    if eq_pin < 0:
        missing.append("equality <connect name='stance_pin' .../>")
    if missing:
        raise ValueError("Your XML is missing required elements:\n" + "\n".join(f"  - {m}" for m in missing))

    # mocap arrays indexed by mocap-id
    pin_mocap_id = int(model.body_mocapid[pin_mocap_body])
    if pin_mocap_id < 0:
        raise ValueError("Body 'pin_mocap' exists but is not mocap='true' in the XML.")

    # init
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if qadr_pelvis_tz is not None:
        data.qpos[qadr_pelvis_tz] = float(args.pelvis_z)
    data.qpos[jidx_leg] = np.array([0.0, -0.22, 0.0], dtype=float)
    mujoco.mj_forward(model, data)

    # phase generator (binary + conf)
    phase_steps = max(1, int(round(float(args.seconds) * float(args.hz))))
    try:
        phase_gen = stream_proxy_phases_from_uci(
            hz=float(args.hz),
            n_steps=int(phase_steps),
            mode=str(args.mode),
            which=str(args.which),
            seed=int(args.uci_seed),
            sleep=False,
        )
    except TypeError:
        phase_gen = stream_proxy_phases_from_uci(
            hz=float(args.hz),
            n_steps=int(phase_steps),
            mode=str(args.mode),
            which=str(args.which),
            seed=int(args.uci_seed),
        )

    # latency buffer (sim-step based)
    latency_steps = int(round((float(args.latency_ms) / 1000.0) / float(args.dt)))
    phase_delay_buf = deque([1] * max(1, latency_steps + 1), maxlen=max(1, latency_steps + 1))
    conf_delay_buf = deque([0.8] * max(1, latency_steps + 1), maxlen=max(1, latency_steps + 1))

    def set_pin_active(active: bool) -> None:
        data.eq_active[eq_pin] = 1 if active else 0

    def move_pin_to_current_foot() -> None:
        p = data.site_xpos[rpin_site].copy()
        data.mocap_pos[pin_mocap_id] = p
        data.mocap_quat[pin_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # Start pinned
    move_pin_to_current_foot()
    set_pin_active(True)
    mujoco.mj_forward(model, data)
    pin_is_active = True

    # viewer
    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        time.sleep(0.05)

    # live plot
    live_plot = bool(args.live_plot)
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        PLOT_WINDOW_SEC = 8.0
        t_buf, raw_buf, app_buf, conf_buf = [], [], [], []
        (l_raw,) = ax[0].plot([], [], label="raw phase")
        (l_app,) = ax[0].plot([], [], "--", label="applied phase")
        ax[0].set_ylim(-0.1, 1.1)
        ax[0].legend(loc="upper right")
        ax[0].set_ylabel("phase")
        (l_conf,) = ax[1].plot([], [], label="confidence")
        ax[1].set_ylim(0.0, 1.05)
        ax[1].legend(loc="upper right")
        ax[1].set_ylabel("conf")
        ax[1].set_xlabel("time [s]")

        PLOT_HZ = 20.0
        plot_every = max(1, int(round(1.0 / (PLOT_HZ * float(args.dt)))))

        def trim(now_t: float):
            while t_buf and (now_t - t_buf[0]) > PLOT_WINDOW_SEC:
                t_buf.pop(0); raw_buf.pop(0); app_buf.pop(0); conf_buf.pop(0)
    else:
        plot_every = 10**9

    # phase ticking schedule
    phase_tick_interval = max(1, int(round(1.0 / (float(args.hz) * float(args.dt)))))
    current_phase = 1
    current_conf = 0.8

    # --- Streamed-phase -> continuous gait phi state ---
    stance_ratio = float(args.stance_ratio)
    stance_ratio = min(max(stance_ratio, 0.05), 0.95)
    gait_hz = float(args.gait_hz)
    gait_hz = max(gait_hz, 1e-3)

    stance_dur = stance_ratio / gait_hz          # seconds
    swing_dur = (1.0 - stance_ratio) / gait_hz   # seconds
    stance_phi0, stance_phi1 = 0.0, stance_ratio
    swing_phi0, swing_phi1 = stance_ratio, 1.0

    seg_phase = 1            # current segment binary phase (stance=1/swing=0)
    seg_t0 = 0.0             # segment start time (sim time)
    phi = 0.0                # continuous phase
    last_good_phi = 0.0      # hold when confidence low

    def phi_from_segment(t_sim: float, seg_is_stance: bool, seg_t0_local: float) -> float:
        if seg_is_stance:
            dur = stance_dur
            a, b = stance_phi0, stance_phi1
        else:
            dur = swing_dur
            a, b = swing_phi0, swing_phi1
        prog = (t_sim - seg_t0_local) / max(dur, 1e-6)
        prog = min(max(prog, 0.0), 1.0)
        return a + (b - a) * prog

    # timing
    n_steps = int(round(float(args.seconds) / float(args.dt)))
    use_realtime = (not args.no_sleep)
    t0_wall = time.perf_counter()

    for step in range(n_steps):
        # streamed phase update
        if step % phase_tick_interval == 0:
            try:
                st = next(phase_gen)
                current_phase = int(st.phase)
                current_conf = float(getattr(st, "confidence", 0.5))
            except StopIteration:
                pass

        # apply latency
        phase_delay_buf.append(current_phase)
        conf_delay_buf.append(current_conf)
        applied_phase = int(phase_delay_buf[0]) if latency_steps > 0 else int(current_phase)
        applied_conf = float(conf_delay_buf[0]) if latency_steps > 0 else float(current_conf)

        now_t = step * float(args.dt)

        # --- Build continuous phi from streamed phase ---
        # Confidence gating: if too low, hold phi (avoids jittery switching)
        if applied_conf < float(args.conf_min):
            phi = last_good_phi
            stance = (seg_phase == 1)
        else:
            # Detect segment change (stance<->swing)
            if applied_phase != seg_phase:
                seg_phase = applied_phase
                seg_t0 = now_t  # reset segment time

                # When we enter stance, also snap pin target to current foot and enable connect
                # When we enter swing, disable connect
                if seg_phase == 1 and (not pin_is_active):
                    move_pin_to_current_foot()
                    set_pin_active(True)
                    pin_is_active = True
                elif seg_phase == 0 and pin_is_active:
                    set_pin_active(False)
                    pin_is_active = False

            stance = (seg_phase == 1)
            phi = phi_from_segment(now_t, stance, seg_t0)
            last_good_phi = phi

        # If not using streamed-phase gait, fall back to time-phase (but you asked streamed-phase)
        if not args.use_stream_phase_gait:
            # old time-based gait
            phi = ((now_t * gait_hz) % 1.0)
            stance = (phi < stance_ratio)

        # stance pin safety (in case user disabled streamed gait but wants pin)
        if stance and (not pin_is_active):
            move_pin_to_current_foot()
            set_pin_active(True)
            pin_is_active = True
        elif (not stance) and pin_is_active:
            set_pin_active(False)
            pin_is_active = False

        # compute leg target from continuous phi
        qref_leg = gait_reference_from_phase(phi)

        # controls
        data.ctrl[:] = 0.0
        if a_pelvis >= 0:
            data.ctrl[a_pelvis] = float(args.pelvis_z)
        data.ctrl[aidx_leg] = clip_ctrl(model, aidx_leg, qref_leg)

        mujoco.mj_step(model, data)

        if args.debug_contacts and step % int(0.2 / float(args.dt)) == 0:
            pz = data.qpos[qadr_pelvis_tz] if qadr_pelvis_tz is not None else float("nan")
            print(
                f"t={now_t:5.2f}s  raw={current_phase} app={applied_phase} "
                f"conf={applied_conf:.2f}  pin={'ON' if pin_is_active else 'OFF'} "
                f"phi={phi:.3f}  pelvis_z={pz:.3f}"
            )

        if viewer is not None:
            viewer.sync()

        if live_plot:
            t_buf.append(now_t)
            raw_buf.append(float(current_phase))
            app_buf.append(float(applied_phase))
            conf_buf.append(float(applied_conf))
            trim(now_t)
            if step % plot_every == 0:
                l_raw.set_data(t_buf, raw_buf)
                l_app.set_data(t_buf, app_buf)
                l_conf.set_data(t_buf, conf_buf)
                ax[0].set_xlim(max(0.0, now_t - 8.0), now_t + 1e-3)
                fig.canvas.draw()
                fig.canvas.flush_events()

        if use_realtime:
            target_wall = t0_wall + now_t
            while True:
                rem = target_wall - time.perf_counter()
                if rem <= 0:
                    break
                time.sleep(min(0.001, rem))

    if viewer is not None:
        viewer.close()
    if live_plot:
        plt.ioff()
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
