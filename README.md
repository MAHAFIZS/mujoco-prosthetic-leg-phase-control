# MuJoCo Prosthetic Leg – Real-Time Phase Control Simulation

A physics-based simulation framework for evaluating real-time prosthetic leg control under noisy and delayed gait phase estimation.

This project demonstrates how a knee–ankle prosthetic system can remain stable when driven by streamed phase signals (e.g., from EMG/IMU sensors), even under latency and uncertainty.

---

## 🎯 Motivation

Robotics is not limited to delivery systems and automation.

Assistive robotics aims to restore mobility and improve quality of life.  
However, prosthetic control systems must operate reliably under:

- Noisy phase classification
- Sensor uncertainty
- Streaming latency
- Real-world contact constraints

This project provides a MuJoCo-based evaluation environment to test robustness of phase-driven prosthetic control strategies.

---

## ⚙️ System Overview

The simulation includes:

- Dummy human model with prosthetic right leg
- Physics-based ground contact
- Real-time phase streaming
- Confidence-based gating
- Latency modeling
- Continuous gait trajectory generation
- Live phase & confidence visualization

The prosthetic hip, knee, and ankle joints track phase-dependent reference trajectories using position control.

---

## 🧠 Key Concepts

### 1️⃣ Streamed Gait Phase
The controller receives phase signals in real-time (binary or continuous).

### 2️⃣ Latency Modeling
Classification delay can be simulated:
```bash
--latency_ms 120
