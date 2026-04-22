"""
simulation.py — FSM integration loop
"""
import numpy as np
from scipy.integrate import solve_ivp
from parameters import (
    LOCK_H_UPPER, BASIN_1_H_INIT, BASIN_2_H_INIT, BASIN_3_H_INIT,
    EPSILON, DT_MAX
)
from dynamics import STATE_NAMES, PHASE_CONFIG, H_sea, system_dynamics

def run_lock_cycle(cd_override=None, valve_time_override=None):
    import parameters as P
    _cd = P.CD; _vt = P.VALVE_OPEN_TIME
    if cd_override is not None:        P.CD = cd_override
    if valve_time_override is not None: P.VALVE_OPEN_TIME = valve_time_override

    y0 = [LOCK_H_UPPER, BASIN_3_H_INIT, BASIN_2_H_INIT, BASIN_1_H_INIT, 0.0]
    t_all, x_all, log = [], [], []
    current_time = 0.0

    for phase_idx, (name, (target_idx, _)) in enumerate(zip(STATE_NAMES, PHASE_CONFIG)):
        t_entry = current_time

        def make_event(tidx):
            def event(t, y):
                h_target = H_sea(t) if tidx == -1 else y[tidx]
                return (y[0] - h_target) - EPSILON
            event.terminal  = True
            event.direction = -1
            return event

        sol = solve_ivp(
            fun=lambda t, y, pi=phase_idx, te=t_entry: system_dynamics(t, y, pi, te),
            t_span=(current_time, current_time + 3600),
            y0=y0,
            method='RK45',
            max_step=DT_MAX,
            events=make_event(target_idx),
        )

        t_all.append(sol.t)
        x_all.append(sol.y)
        log.append((name, t_entry, sol.t[-1]))
        current_time = sol.t[-1]
        y0 = list(sol.y[:, -1])
        y0[4] = 0.0

    P.CD = _cd; P.VALVE_OPEN_TIME = _vt
    return np.concatenate(t_all), np.hstack(x_all), log


if __name__ == '__main__':
    from parameters import BASIN_1_H_INIT, BASIN_2_H_INIT, BASIN_3_H_INIT
    t, x, log = run_lock_cycle()
    print(f"Total time: {t[-1]/60:.1f} min\n")
    for name, ti, to in log:
        print(f"  {name:<38} {ti/60:5.1f} → {to/60:5.1f} min  ({(to-ti)/60:.1f} min)")
    print(f"\nPeak Q:    {x[4].max():.1f} m³/s")
    print(f"Final H_L: {x[0,-1]:.2f} m")
    print(f"\nFinal basin levels:")
    print(f"  H_B3={x[1,-1]:.2f}m (was {BASIN_3_H_INIT}m)")
    print(f"  H_B2={x[2,-1]:.2f}m (was {BASIN_2_H_INIT}m)")
    print(f"  H_B1={x[3,-1]:.2f}m (was {BASIN_1_H_INIT}m)")