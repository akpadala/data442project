"""
simulation.py
FSM-driven integration loop. Runs solve_ivp for each FSM state,
checks transition conditions, switches dynamics, and accumulates results.
"""

import numpy as np
from scipy.integrate import solve_ivp

from parameters import (
    LOCK_H_UPPER, BASIN_1_H_INIT, BASIN_2_H_INIT, BASIN_3_H_INIT,
    EPSILON, T_MAX, DT_MAX
)
from dynamics import (
    VESSEL_ENTRY, GATE_CLOSING,
    WSB_DRAIN_1, WSB_DRAIN_2, WSB_DRAIN_3,
    FINAL_DRAIN, GATE_OPENING, VESSEL_EXIT,
    STATE_NAMES, ODE_MAP, H_sea
)

# ── Transition condition ──────────────────────────────────────────────────────

def check_transition(state, x):
    """
    Returns the next FSM state if the transition condition is met,
    otherwise returns the current state.

    Transitions trigger when the active water pair equalizes: |ΔH| < ε
    """
    H_L, H_B3, H_B2, H_B1, Q = x

    if state == WSB_DRAIN_1 and abs(H_L - H_B3) < EPSILON:
        return WSB_DRAIN_2
    if state == WSB_DRAIN_2 and abs(H_L - H_B2) < EPSILON:
        return WSB_DRAIN_3
    if state == WSB_DRAIN_3 and abs(H_L - H_B1) < EPSILON:
        return FINAL_DRAIN
    if state == FINAL_DRAIN and abs(H_L - H_sea(0)) < EPSILON:
        # Simplified: transition when lock is near mean sea level
        return GATE_OPENING

    return state  # no transition yet

# ── Single lock cycle ─────────────────────────────────────────────────────────

def run_lock_cycle(t_start=0.0, x0=None):
    """
    Simulate one complete lock lowering cycle:
    WSB_DRAIN_1 → WSB_DRAIN_2 → WSB_DRAIN_3 → FINAL_DRAIN

    Returns:
        t_all : array of time points (s)
        x_all : array of state vectors, shape (5, len(t_all))
        log   : list of (state_name, t_entry, t_exit) for each phase
    """
    if x0 is None:
        x0 = [
            LOCK_H_UPPER,      # H_L  — lock starts full
            BASIN_3_H_INIT,    # H_B3
            BASIN_2_H_INIT,    # H_B2
            BASIN_1_H_INIT,    # H_B1
            0.0,               # Q    — flow starts at zero
        ]

    # Active ODE states in sequence
    drain_states = [WSB_DRAIN_1, WSB_DRAIN_2, WSB_DRAIN_3, FINAL_DRAIN]

    t_all = []
    x_all = []
    log   = []

    t_current = t_start
    x_current = np.array(x0, dtype=float)

    for state in drain_states:
        t_state_entry = t_current
        ode_func = ODE_MAP[state]

        print(f"  [{STATE_NAMES[state]}]  t={t_current:.1f}s  "
              f"H_L={x_current[0]:.2f}m  Q={x_current[4]:.1f}m³/s")

        # Solve until transition condition or time runs out
        t_end = t_start + T_MAX
        max_steps = int((t_end - t_current) / DT_MAX) + 1

        sol = solve_ivp(
            fun=lambda t, x: ode_func(t, x, t_state_entry),
            t_span=(t_current, t_end),
            y0=x_current,
            method='RK45',
            max_step=DT_MAX,
            dense_output=True,
            events=_make_transition_event(state),   # scipy event detection
        )

        # Collect results up to transition event
        t_seg = sol.t
        x_seg = sol.y  # shape (5, n_points)

        t_all.append(t_seg)
        x_all.append(x_seg)

        # Find exit state
        if sol.t_events and len(sol.t_events[0]) > 0:
            t_exit = sol.t_events[0][0]
            x_exit = sol.sol(t_exit)
        else:
            t_exit = sol.t[-1]
            x_exit = sol.y[:, -1]

        log.append((STATE_NAMES[state], t_state_entry, t_exit))

        t_current = t_exit
        x_current = x_exit
        x_current[4] = 0.0  # reset Q to zero at valve close (new state starts fresh)

        if t_current >= t_start + T_MAX:
            break

    t_all = np.concatenate(t_all)
    x_all = np.concatenate(x_all, axis=1)

    return t_all, x_all, log


def _make_transition_event(state):
    """
    Returns a scipy event function that triggers when the active
    water pair equalizes (|ΔH| < EPSILON).
    """
    def event(t, x):
        H_L, H_B3, H_B2, H_B1, Q = x
        if state == WSB_DRAIN_1:
            return abs(H_L - H_B3) - EPSILON
        if state == WSB_DRAIN_2:
            return abs(H_L - H_B2) - EPSILON
        if state == WSB_DRAIN_3:
            return abs(H_L - H_B1) - EPSILON
        if state == FINAL_DRAIN:
            return abs(H_L - H_sea(t)) - EPSILON
        return 1.0  # never triggers for other states

    event.terminal  = True   # stop integration when event fires
    event.direction = -1     # only trigger when |ΔH| is decreasing toward ε
    return event


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Panama Canal lock cycle simulation...")
    t, x, log = run_lock_cycle()

    print("\nPhase log:")
    for name, t_in, t_out in log:
        print(f"  {name:<35} {t_in:7.1f}s → {t_out:7.1f}s  "
              f"(duration: {t_out - t_in:.1f}s)")

    print(f"\nFinal water levels:")
    print(f"  H_L  = {x[0, -1]:.3f} m")
    print(f"  H_B3 = {x[1, -1]:.3f} m")
    print(f"  H_B2 = {x[2, -1]:.3f} m")
    print(f"  H_B1 = {x[3, -1]:.3f} m")
    print(f"  Q    = {x[4, -1]:.3f} m³/s")
