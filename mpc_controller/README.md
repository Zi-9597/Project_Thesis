# 🚗 LPV–MPC Controller (ROS 2 + OSQP)

## Overview

This package implements a Linear Parameter Varying Model Predictive Controller (LPV–MPC) 
for autonomous vehicle control using ROS 2.

The controller receives real-time vehicle states from CAN topics and publishes steering
and acceleration commands to the vehicle.

The optimization problem is formulated as a Quadratic Program (QP) and solved using OSQP.

---

## 📐 Control Model

State vector:

    x = [ Vx, Vy, ω ]ᵀ

Input vector:

    u = [ δ, ax ]ᵀ

Discrete LPV model:

    xₖ₊₁ = A(ρₖ) xₖ + B(ρₖ) uₖ

where ρₖ are scheduling variables (typically Vx).

---

## 🎯 Optimization Problem

At each MPC step, the following QP is solved:

    minimize Σ (x - x_ref)' Q (x - x_ref) + u' R u

subject to:

    xₖ₊₁ = Aₖ xₖ + Bₖ uₖ
    x_min ≤ xₖ ≤ x_max
    u_min ≤ uₖ ≤ u_max

---

## ⚙️ Real-Time Structure

| Component         | Frequency |
|------------------|-----------|
| Sensor updates   | 100 Hz    |
| Command publish  | 100 Hz    |
| MPC solve        | ~30 Hz    |

The input is held constant between MPC updates (Zero-Order Hold).

---

## 📦 File Structure
class_mpc.py # QP formulation and OSQP solver
├── dynamics_command.py # ROS 2 node (subscriptions + publishing)
├── utils_system.py # LPV model & system utilities
├── README.md


## 🧰 Dependencies

Install required libraries:

    pip install numpy
    pip install scipy
    pip install osqp

ROS 2 dependencies:
    rclpy
    can_zoe_msgs


## 🛡 Current Version

- Nominal LPV–MPC
- OSQP-based QP solver
- No tube/zonotope tightening (future extension)

---