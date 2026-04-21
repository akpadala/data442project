import numpy as np
from parameters import (
    G, RHO, INERTANCE, RESISTANCE, LOCK_AREA, BASIN_AREA,
    TIDE_AMPLITUDE, TIDE_OMEGA, TIDE_MEAN
)

# ── FSM Constants for analysis.py compatibility ──────────────────────────────
WSB_DRAIN_1 = 'WSB drain 1 (lock→basin 3)'
WSB_DRAIN_2 = 'WSB drain 2 (lock→basin 2)'
WSB_DRAIN_3 = 'WSB drain 3 (lock→basin 1)'
FINAL_DRAIN = 'Final drain (lock→sea)'
STATE_NAMES = [WSB_DRAIN_1, WSB_DRAIN_2, WSB_DRAIN_3, FINAL_DRAIN]

def H_sea(t):
    """Calculates sea level based on tidal parameters."""
    return TIDE_MEAN + TIDE_AMPLITUDE * np.sin(TIDE_OMEGA * t)

def system_dynamics(t, y, active_basin_idx):
    """
    y = [h_lock, h_b3, h_b2, h_b1, Q]
    active_basin_idx: 3 (Basin 3), 2 (Basin 2), 1 (Basin 1), 0 (Sea)
    """
    h_l, h_b3, h_b2, h_b1, Q = y
    
    # Map active basin level
    basins = {0: H_sea(t), 1: h_b1, 2: h_b2, 3: h_b3}
    h_target = basins[active_basin_idx]
    
    # Delta Head
    delta_h = h_l - h_target
    
    # Water Hammer Equation: L * dQ/dt + R * Q = Delta_H
    # dQ/dt = (Delta_H - R * Q) / L
    dQdt = (delta_h - (RESISTANCE * Q)) / INERTANCE
    
    # Mass Balance: dh/dt = -Q/Area
    dhl_dt = -Q / LOCK_AREA
    
    # Derivative array initialization
    dydt = [dhl_dt, 0.0, 0.0, 0.0, dQdt]
    
    # Update target basin derivative
    if active_basin_idx != 0:
        dydt[active_basin_idx] = Q / BASIN_AREA
        
    return dydt