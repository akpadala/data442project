"""
dynamics.py
ODE right-hand sides for each FSM state, valve actuation curve,
and tidal forcing function.

State vector x = [H_L, H_B3, H_B2, H_B1, Q]
  H_L  : lock chamber water level (m)
  H_B3 : basin 3 level — highest (m)
  H_B2 : basin 2 level — mid (m)
  H_B1 : basin 1 level — lowest (m)
  Q    : volumetric flow rate through active culvert (m³/s)
"""

import numpy as np
from parameters import (
    G, RHO, CD, INERTANCE, RESISTANCE,
    LOCK_AREA, BASIN_AREA,
    A_VALVE_MAX, VALVE_OPEN_TIME,
    TIDE_AMPLITUDE, TIDE_OMEGA, TIDE_PHASE, TIDE_MEAN
)

# ── FSM state labels ──────────────────────────────────────────────────────────
VESSEL_ENTRY  = 0
GATE_CLOSING  = 1
WSB_DRAIN_1   = 2   # lock → basin 3  (highest basin, first to fill)
WSB_DRAIN_2   = 3   # lock → basin 2
WSB_DRAIN_3   = 4   # lock → basin 1  (lowest basin, last to fill)
FINAL_DRAIN   = 5   # lock → tidal sea (water lost)
GATE_OPENING  = 6
VESSEL_EXIT   = 7

STATE_NAMES = {
    VESSEL_ENTRY : "Vessel entry",
    GATE_CLOSING : "Gate closing",
    WSB_DRAIN_1  : "WSB drain 1 (lock→basin 3)",
    WSB_DRAIN_2  : "WSB drain 2 (lock→basin 2)",
    WSB_DRAIN_3  : "WSB drain 3 (lock→basin 1)",
    FINAL_DRAIN  : "Final drain (lock→sea)",
    GATE_OPENING : "Gate opening",
    VESSEL_EXIT  : "Vessel exit",
}

# ── Valve actuation curve ─────────────────────────────────────────────────────
def valve_area(t, t_state_entry):
    """
    Valve opens linearly from 0 to A_VALVE_MAX over VALVE_OPEN_TIME seconds.
    t_state_entry: time at which the current FSM state was entered.
    """
    elapsed = t - t_state_entry
    frac = np.clip(elapsed / VALVE_OPEN_TIME, 0.0, 1.0)
    return A_VALVE_MAX * frac

def valve_area_derivative(t, t_state_entry):
    """dA_v/dt — constant ramp rate, zero before open and after fully open."""
    elapsed = t - t_state_entry
    if 0.0 <= elapsed <= VALVE_OPEN_TIME:
        return A_VALVE_MAX / VALVE_OPEN_TIME
    return 0.0

# ── Tidal forcing ─────────────────────────────────────────────────────────────
def H_sea(t):
    """Pacific tidal sea level as a function of time (m)."""
    return TIDE_MEAN + TIDE_AMPLITUDE * np.sin(TIDE_OMEGA * t + TIDE_PHASE)

# ── ODE right-hand sides ──────────────────────────────────────────────────────

def _flow_ode(Q, delta_H, A_v, A_v_dot):
    """
    Inertance-based flow ODE (water hammer model):
        L * dQ/dt = ρg * ΔH - R * Q - (A_v_dot / A_v) * L * Q

    Returns dQ/dt.
    Falls back to Torricelli instantaneous flow when A_v ≈ 0 (valve closed).
    """
    if A_v < 1e-6:
        return 0.0
    # Effective head accounting for valve restriction
    # Q_torricelli = Cd * A_v * sqrt(2g * |ΔH|)
    # Driving pressure = ρg * ΔH, back-pressure from inertia/friction
    inertia_correction = (A_v_dot / A_v) * INERTANCE * Q if A_v > 1e-6 else 0.0
    dQ_dt = (RHO * G * delta_H - RESISTANCE * Q - inertia_correction) / INERTANCE
    return dQ_dt

def ode_wsb_drain_1(t, x, t_state_entry):
    """
    WSB_DRAIN_1: lock drains into basin 3.
    x = [H_L, H_B3, H_B2, H_B1, Q]
    Active pair: H_L, H_B3, Q
    H_B2, H_B1 frozen (valves closed).
    """
    H_L, H_B3, H_B2, H_B1, Q = x
    A_v     = valve_area(t, t_state_entry)
    A_v_dot = valve_area_derivative(t, t_state_entry)
    delta_H = H_L - H_B3

    dQ_dt   = _flow_ode(Q, delta_H, A_v, A_v_dot)
    dHL_dt  = -Q / LOCK_AREA
    dHB3_dt = +Q / BASIN_AREA
    dHB2_dt = 0.0
    dHB1_dt = 0.0

    return [dHL_dt, dHB3_dt, dHB2_dt, dHB1_dt, dQ_dt]

def ode_wsb_drain_2(t, x, t_state_entry):
    """
    WSB_DRAIN_2: lock drains into basin 2.
    Active pair: H_L, H_B2, Q
    """
    H_L, H_B3, H_B2, H_B1, Q = x
    A_v     = valve_area(t, t_state_entry)
    A_v_dot = valve_area_derivative(t, t_state_entry)
    delta_H = H_L - H_B2

    dQ_dt   = _flow_ode(Q, delta_H, A_v, A_v_dot)
    dHL_dt  = -Q / LOCK_AREA
    dHB3_dt = 0.0
    dHB2_dt = +Q / BASIN_AREA
    dHB1_dt = 0.0

    return [dHL_dt, dHB3_dt, dHB2_dt, dHB1_dt, dQ_dt]

def ode_wsb_drain_3(t, x, t_state_entry):
    """
    WSB_DRAIN_3: lock drains into basin 1.
    Active pair: H_L, H_B1, Q
    """
    H_L, H_B3, H_B2, H_B1, Q = x
    A_v     = valve_area(t, t_state_entry)
    A_v_dot = valve_area_derivative(t, t_state_entry)
    delta_H = H_L - H_B1

    dQ_dt   = _flow_ode(Q, delta_H, A_v, A_v_dot)
    dHL_dt  = -Q / LOCK_AREA
    dHB3_dt = 0.0
    dHB2_dt = 0.0
    dHB1_dt = +Q / BASIN_AREA

    return [dHL_dt, dHB3_dt, dHB2_dt, dHB1_dt, dQ_dt]

def ode_final_drain(t, x, t_state_entry):
    """
    FINAL_DRAIN: lock drains to tidal sea level.
    This is the 'bumpy road' state — H_sea(t) is the sinusoidal forcing.
    Active pair: H_L, H_sea(t), Q
    All basins frozen.
    """
    H_L, H_B3, H_B2, H_B1, Q = x
    A_v     = valve_area(t, t_state_entry)
    A_v_dot = valve_area_derivative(t, t_state_entry)
    delta_H = H_L - H_sea(t)      # ← tidal forcing enters here

    dQ_dt   = _flow_ode(Q, delta_H, A_v, A_v_dot)
    dHL_dt  = -Q / LOCK_AREA
    dHB3_dt = 0.0
    dHB2_dt = 0.0
    dHB1_dt = 0.0

    return [dHL_dt, dHB3_dt, dHB2_dt, dHB1_dt, dQ_dt]

# ── Dispatch table: FSM state → ODE function ──────────────────────────────────
ODE_MAP = {
    WSB_DRAIN_1 : ode_wsb_drain_1,
    WSB_DRAIN_2 : ode_wsb_drain_2,
    WSB_DRAIN_3 : ode_wsb_drain_3,
    FINAL_DRAIN : ode_final_drain,
}
