"""
dynamics.py
State vector y = [h_l, h_b3, h_b2, h_b1, Q]
                  0     1     2     3     4

Each phase specifies:
  - y_idx_target : index in y of the target water body (1=B3, 2=B2, 3=B1, -1=sea)
  - y_idx_fill   : index in y to fill (same as target, or -1 for sea)
"""
import numpy as np
from parameters import (
    G, RHO, INERTANCE, RESISTANCE,
    LOCK_AREA, BASIN_AREA,
    A_VALVE_MAX, VALVE_OPEN_TIME,
    TIDE_AMPLITUDE, TIDE_OMEGA, TIDE_MEAN, TIDE_PHASE,
    CD
)

WSB_DRAIN_1 = 'WSB drain 1 (lock→basin 3)'
WSB_DRAIN_2 = 'WSB drain 2 (lock→basin 2)'
WSB_DRAIN_3 = 'WSB drain 3 (lock→basin 1)'
FINAL_DRAIN = 'Final drain (lock→sea)'
STATE_NAMES = [WSB_DRAIN_1, WSB_DRAIN_2, WSB_DRAIN_3, FINAL_DRAIN]

# For each phase: (y_index_of_target_basin, y_index_to_fill)
# Sea phase uses sentinel -1
PHASE_CONFIG = [
    (1, 1),   # drain 1: target=y[1]=h_b3, fill y[1]
    (2, 2),   # drain 2: target=y[2]=h_b2, fill y[2]
    (3, 3),   # drain 3: target=y[3]=h_b1, fill y[3]
    (-1, -1), # final:   target=sea,        fill nothing
]

def H_sea(t):
    return TIDE_MEAN + TIDE_AMPLITUDE * np.sin(TIDE_OMEGA * t + TIDE_PHASE)

def valve_area(t, t_entry):
    elapsed = t - t_entry
    return A_VALVE_MAX * float(np.clip(elapsed / VALVE_OPEN_TIME, 0.0, 1.0))

def system_dynamics(t, y, phase_idx, t_entry):
    """
    phase_idx: 0,1,2,3 corresponding to PHASE_CONFIG
    """
    h_l, h_b3, h_b2, h_b1, Q = y
    target_idx, fill_idx = PHASE_CONFIG[phase_idx]

    # Target level
    if target_idx == -1:
        h_target = H_sea(t)
    else:
        h_target = y[target_idx]

    delta_h = max(h_l - h_target, 0.0)
    Av = valve_area(t, t_entry)

    # Water hammer ODE (FIXED: RHO*G scaling, valve ramp)
    driving  = RHO * G * delta_h * CD * (Av / A_VALVE_MAX)
    friction = RESISTANCE * Q
    dQdt     = (driving - friction) / INERTANCE

    dhl_dt = -Q / LOCK_AREA

    dydt = [dhl_dt, 0.0, 0.0, 0.0, dQdt]
    if fill_idx != -1:
        dydt[fill_idx] = Q / BASIN_AREA

    return dydt