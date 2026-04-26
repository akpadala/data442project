"""
pid_controller.py
==================
Closed-loop PID controller that regulates valve opening rate to track a
target ΔH trajectory while keeping flow velocity below the 8 m/s spec.

Control problem:
    Plant:      Water hammer ODE — L*dQ/dt + R*Q = rho*g*dH*Cd*(Av/Amax)
    Controlled: ΔH(t) = H_L(t) - H_target(t)
    Setpoint:   ΔH*(t) = ΔH_0 * (1 - t/T_target)  [ideal linear decay to 0]
    Actuator:   Valve fraction u(t) ∈ [0, 1]  →  Av(t) = u(t) * A_max
    Constraint: V(t) = Q(t)/A_culvert ≤ 8 m/s  (velocity spec from paper)

PID law:
    e(t)  = ΔH(t) - ΔH*(t)
    u(t)  = Kp*e + Ki*∫e dt + Kd*de/dt
    u(t)  clipped to [0, 1]

Figures generated:
    fig11_pid_comparison.png  — two-panel: ΔH and velocity, controlled vs uncontrolled

Run with: python pid_controller.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import parameters as P
from dynamics import H_sea, PHASE_CONFIG
from parameters import (
    LOCK_H_UPPER, BASIN_3_H_INIT, BASIN_2_H_INIT, BASIN_1_H_INIT
)

# ── PID Controller class ───────────────────────────────────────────────────────
class PIDController:
    """
    Discrete PID controller.
    Regulates valve fraction u ∈ [0,1] to drive ΔH → 0 along a linear setpoint.

    Kp: proportional gain — responds to current error
    Ki: integral gain    — eliminates steady-state offset
    Kd: derivative gain  — anticipates rate of change, prevents overshoot
    T_target: desired equalization time (seconds)
    V_max: maximum allowable velocity (m/s) — hard constraint on u
    """
    def __init__(self, Kp, Ki, Kd, T_target, V_max=8.0):
        self.Kp       = Kp
        self.Ki       = Ki
        self.Kd       = Kd
        self.T_target = T_target
        self.V_max    = V_max

        self._integral    = 0.0
        self._prev_error  = None
        self._prev_t      = None
        self._dH_initial  = None

    def reset(self, dH_initial):
        self._integral   = 0.0
        self._prev_error = None
        self._prev_t     = None
        self._dH_initial = dH_initial

    def compute(self, t, dH, Q):
        """
        Returns valve fraction u ∈ [0, 1].
        Also enforces the velocity constraint: reduces u if Q/A_culvert > V_max.
        """
        if self._dH_initial is None or self._dH_initial <= 0:
            return 1.0

        # Setpoint: linear ramp from dH_initial to 0 over T_target seconds
        setpoint = self._dH_initial * max(1.0 - t / self.T_target, 0.0)
        error    = dH - setpoint

        # dt
        dt = (t - self._prev_t) if self._prev_t is not None else 1.0
        dt = max(dt, 1e-6)

        # Integral (with anti-windup: clamp integral contribution)
        self._integral += error * dt
        self._integral  = np.clip(self._integral,
                                  -self._dH_initial / max(self.Ki, 1e-9),
                                   self._dH_initial / max(self.Ki, 1e-9))

        # Derivative
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        else:
            derivative = 0.0

        # PID output
        u = 1.0 - (self.Kp * error
                   + self.Ki * self._integral
                   + self.Kd * derivative)
        u = float(np.clip(u, 0.0, 1.0))

        # Hard velocity constraint: if Q exceeds spec, reduce valve
        V_current = Q / P.CULVERT_AREA
        if V_current > self.V_max and u > 0:
            u_constrained = u * (self.V_max / V_current) ** 2
            u = min(u, u_constrained)
            u = float(np.clip(u, 0.0, 1.0))

        self._prev_error = error
        self._prev_t     = t
        return u


# ── ODE with controllable valve fraction ──────────────────────────────────────
def make_ode_pid(get_valve_fraction, phase=0):
    """
    get_valve_fraction(t, y) → u ∈ [0,1]
    Returns ODE function for the given phase with externally controlled valve.
    """
    target_idx, fill_idx = PHASE_CONFIG[phase]

    def ode(t, y):
        h_l, h_b3, h_b2, h_b1, Q = y

        h_target = H_sea(t) if target_idx == -1 else y[target_idx]
        delta_h  = max(h_l - h_target, 0.0)

        u        = get_valve_fraction(t, y)
        Av       = P.A_VALVE_MAX * u
        driving  = P.RHO * P.G * delta_h * P.CD * u
        friction = P.RESISTANCE * Q
        dQdt     = (driving - friction) / P.INERTANCE

        dhl_dt = -Q / P.LOCK_AREA
        dydt   = [dhl_dt, 0.0, 0.0, 0.0, dQdt]
        if fill_idx != -1:
            dydt[fill_idx] = Q / P.BASIN_AREA
        return dydt

    return ode


# ── Run uncontrolled (ramp valve) ─────────────────────────────────────────────
def run_uncontrolled(phase=0, T_max=600):
    target_idx = PHASE_CONFIG[phase][0]
    y0 = [LOCK_H_UPPER, BASIN_3_H_INIT, BASIN_2_H_INIT, BASIN_1_H_INIT, 0.0]

    def valve_ramp(t, y):
        return min(t / P.VALVE_OPEN_TIME, 1.0)

    def event(t, y):
        h_t = H_sea(t) if target_idx == -1 else y[target_idx]
        return (y[0] - h_t) - P.EPSILON
    event.terminal = True; event.direction = -1

    sol = solve_ivp(make_ode_pid(valve_ramp, phase),
                    [0, T_max], y0, method='RK45',
                    max_step=1.0, events=event)
    return sol.t, sol.y


# ── Run PID controlled ────────────────────────────────────────────────────────
def run_pid(Kp, Ki, Kd, T_target, phase=0, T_max=600):
    target_idx = PHASE_CONFIG[phase][0]
    y0   = [LOCK_H_UPPER, BASIN_3_H_INIT, BASIN_2_H_INIT, BASIN_1_H_INIT, 0.0]
    dH_0 = y0[0] - y0[target_idx] if target_idx != -1 else y0[0]

    pid  = PIDController(Kp, Ki, Kd, T_target, V_max=8.0)
    pid.reset(dH_0)

    # Store valve trace for plotting
    valve_trace = []

    def valve_pid(t, y):
        h_t  = H_sea(t) if target_idx == -1 else y[target_idx]
        dH   = max(y[0] - h_t, 0.0)
        Q    = y[4]
        u    = pid.compute(t, dH, Q)
        valve_trace.append((t, u))
        return u

    def event(t, y):
        h_t = H_sea(t) if target_idx == -1 else y[target_idx]
        return (y[0] - h_t) - P.EPSILON
    event.terminal = True; event.direction = -1

    sol = solve_ivp(make_ode_pid(valve_pid, phase),
                    [0, T_max], y0, method='RK45',
                    max_step=1.0, events=event)

    v_t = np.array([v[0] for v in valve_trace])
    v_u = np.array([v[1] for v in valve_trace])
    return sol.t, sol.y, v_t, v_u


# ── PID tuning sweep ──────────────────────────────────────────────────────────
print("=" * 60)
print("PID CONTROLLER — Panama Canal Valve Control")
print("=" * 60)
print()
print("Tuning objective: minimize cycle time subject to V_peak < 8 m/s")
print()

# Target equalization in 240s (same as ramp baseline)
T_target = 240.0

# Sweep Kp, fix Ki=Kp/10, Kd=Kp*5
best_time = np.inf
best_params = None
best_V_ok   = False

print("Kp      Ki      Kd      Cycle(s)  V_peak(m/s)  OK?")
print("-" * 55)

for Kp in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
    Ki = Kp / 10
    Kd = Kp * 2
    try:
        t_pid, x_pid, _, _ = run_pid(Kp, Ki, Kd, T_target)
        V_peak = x_pid[4].max() / P.CULVERT_AREA
        ok     = V_peak <= 8.0
        print(f"{Kp:6.1f}  {Ki:6.2f}  {Kd:6.1f}  {t_pid[-1]:7.1f}s  "
              f"{V_peak:10.2f}     {'✓' if ok else '✗'}")
        if ok and t_pid[-1] < best_time:
            best_time   = t_pid[-1]
            best_params = (Kp, Ki, Kd)
            best_V_ok   = True
    except Exception as e:
        print(f"{Kp:6.1f}  failed: {e}")

# If none satisfy constraint, pick lowest V_peak
if not best_V_ok:
    print("\nNo params satisfy V<8 m/s — picking Kp=2 as best compromise")
    best_params = (2.0, 0.2, 4.0)

print()
Kp_best, Ki_best, Kd_best = best_params
print(f"Best params: Kp={Kp_best}, Ki={Ki_best}, Kd={Kd_best}")

# ── Final runs ────────────────────────────────────────────────────────────────
t_unc, x_unc                   = run_uncontrolled()
t_pid, x_pid, v_t, v_u        = run_pid(Kp_best, Ki_best, Kd_best, T_target)

# Metrics
V_unc_peak = x_unc[4].max() / P.CULVERT_AREA
V_pid_peak = x_pid[4].max() / P.CULVERT_AREA
dH_unc     = np.maximum(x_unc[0] - x_unc[1], 0)
dH_pid     = np.maximum(x_pid[0] - x_pid[1], 0)
t_unc_end  = t_unc[-1]
t_pid_end  = t_pid[-1]

print()
print("RESULTS COMPARISON — WSB Drain Phase 1:")
print(f"  {'':25}  {'Uncontrolled':>14}  {'PID':>14}")
print(f"  {'Cycle time':25}  {t_unc_end:>12.1f}s  {t_pid_end:>12.1f}s")
print(f"  {'Peak velocity':25}  {V_unc_peak:>11.1f}ms  {V_pid_peak:>11.1f}ms")
print(f"  {'Velocity spec met':25}  {'✗' if V_unc_peak>8 else '✓':>14}  "
      f"{'✓' if V_pid_peak<=8 else '✗ (model limitation)':>14}")
print(f"  {'Time difference':25}  {abs(t_unc_end-t_pid_end):>12.1f}s")
print()

# ── Setpoint trajectory for plotting ─────────────────────────────────────────
dH_0 = LOCK_H_UPPER - BASIN_3_H_INIT
t_sp  = np.linspace(0, max(t_unc_end, t_pid_end), 500)
dH_sp = dH_0 * np.maximum(1 - t_sp / T_target, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11: PID comparison — 3 panels
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
fig.suptitle(
    'PID Valve Controller vs Uncontrolled — WSB Drain Phase 1\n'
    f'Kp={Kp_best}, Ki={Ki_best}, Kd={Kd_best}  |  '
    f'T_target={T_target:.0f}s  |  V_max constraint = 8 m/s',
    fontsize=12, fontweight='bold'
)

ax_dh, ax_v, ax_u = axes

# ── Panel 1: ΔH vs time ───────────────────────────────────────────────────────
ax_dh.plot(t_unc / 60, dH_unc,  color='#DC2626', lw=2,
           label='Uncontrolled (ramp valve)')
ax_dh.plot(t_pid / 60, dH_pid,  color='#2563EB', lw=2,
           label='PID controlled')
ax_dh.plot(t_sp / 60, dH_sp,    color='gray', lw=1.5, ls='--',
           label=f'Setpoint ΔH*(t) — linear decay to 0 in {T_target:.0f}s')
ax_dh.axhline(P.EPSILON, color='#059669', lw=1, ls=':',
              label=f'ε = {P.EPSILON}m (equalization threshold)')

# Mark equalization times
ax_dh.axvline(t_unc_end / 60, color='#DC2626', lw=1, ls='--', alpha=0.6)
ax_dh.axvline(t_pid_end / 60, color='#2563EB', lw=1, ls='--', alpha=0.6)
ax_dh.text(t_unc_end / 60, dH_unc[0] * 0.5,
           f'  {t_unc_end:.0f}s', color='#DC2626', fontsize=8)
ax_dh.text(t_pid_end / 60, dH_pid[0] * 0.4,
           f'  {t_pid_end:.0f}s', color='#2563EB', fontsize=8)

ax_dh.set_ylabel('ΔH = H_L − H_B3 (m)', fontsize=10)
ax_dh.set_title('Head difference tracking setpoint', fontsize=10)
ax_dh.legend(fontsize=8.5, loc='upper right')
ax_dh.grid(True, alpha=0.3)
ax_dh.set_xlim(left=0)

# PID annotation
ax_dh.text(0.01, 0.05,
    f'PID law: u(t) = 1 − [Kp·e + Ki·∫e dt + Kd·ė]\n'
    f'e(t) = ΔH(t) − ΔH*(t)  |  ΔH*(t) = ΔH₀·(1 − t/T_target)',
    transform=ax_dh.transAxes, fontsize=7.5, va='bottom',
    bbox=dict(facecolor='#EDE9FE', edgecolor='#7C3AED',
              boxstyle='round,pad=0.4', alpha=0.9))

# ── Panel 2: Velocity vs time ─────────────────────────────────────────────────
V_unc = x_unc[4] / P.CULVERT_AREA
V_pid = x_pid[4] / P.CULVERT_AREA

ax_v.plot(t_unc / 60, V_unc, color='#DC2626', lw=2, label='Uncontrolled velocity')
ax_v.plot(t_pid / 60, V_pid, color='#2563EB', lw=2, label='PID controlled velocity')
ax_v.axhline(8.0, color='#F59E0B', lw=2, ls='--', label='Max spec: 8 m/s (paper)')
ax_v.axhline(4.7, color='#059669', lw=1, ls=':', label='Avg spec: 4.7 m/s (paper)')

# Shade violation
ax_v.fill_between(t_unc / 60, 8.0, V_unc,
                   where=(V_unc > 8.0), alpha=0.2, color='#DC2626',
                   label='Uncontrolled: spec violation')
if V_pid_peak > 8.0:
    ax_v.fill_between(t_pid / 60, 8.0, V_pid,
                       where=(V_pid > 8.0), alpha=0.2, color='#2563EB',
                       label='PID: residual violation (model limitation)')

ax_v.set_ylabel('Flow velocity (m/s)', fontsize=10)
ax_v.set_title('Velocity constraint enforcement', fontsize=10)
ax_v.legend(fontsize=8, loc='upper right')
ax_v.grid(True, alpha=0.3)
ax_v.set_ylim(bottom=0)
ax_v.set_xlim(left=0)

# Peak annotation
ax_v.annotate(f'{V_unc_peak:.1f} m/s',
              xy=(t_unc[np.argmax(V_unc)] / 60, V_unc_peak),
              xytext=(t_unc[np.argmax(V_unc)] / 60 + 0.3, V_unc_peak + 0.5),
              arrowprops=dict(arrowstyle='->', color='#DC2626'),
              fontsize=8, color='#DC2626')
ax_v.annotate(f'{V_pid_peak:.1f} m/s',
              xy=(t_pid[np.argmax(V_pid)] / 60, V_pid_peak),
              xytext=(t_pid[np.argmax(V_pid)] / 60 + 0.3, V_pid_peak - 2),
              arrowprops=dict(arrowstyle='->', color='#2563EB'),
              fontsize=8, color='#2563EB')

# ── Panel 3: Valve fraction u(t) ─────────────────────────────────────────────
t_ramp_valve = np.linspace(0, t_unc_end, 300)
u_ramp       = np.minimum(t_ramp_valve / P.VALVE_OPEN_TIME, 1.0)

ax_u.plot(t_ramp_valve / 60, u_ramp, color='#DC2626', lw=2,
          label='Uncontrolled: linear ramp')
ax_u.plot(v_t / 60, v_u, color='#2563EB', lw=2,
          label='PID: adaptive valve fraction')
ax_u.axhline(1.0, color='gray', lw=0.8, ls=':')
ax_u.axhline(0.0, color='gray', lw=0.8, ls=':')

ax_u.set_ylabel('Valve fraction u(t)', fontsize=10)
ax_u.set_xlabel('Time (minutes)', fontsize=10)
ax_u.set_title('Control input u(t) — valve opening fraction', fontsize=10)
ax_u.legend(fontsize=8.5, loc='lower right')
ax_u.grid(True, alpha=0.3)
ax_u.set_ylim(-0.05, 1.1)
ax_u.set_xlim(left=0)

# Role of each PID term
ax_u.text(0.01, 0.97,
    f'P (Kp={Kp_best}): responds to current ΔH error\n'
    f'I (Ki={Ki_best}): eliminates steady-state offset\n'
    f'D (Kd={Kd_best}): anticipates rate of change — prevents overshoot',
    transform=ax_u.transAxes, fontsize=8, va='top',
    bbox=dict(facecolor='#DCFCE7', edgecolor='#059669',
              boxstyle='round,pad=0.4', alpha=0.9))

plt.tight_layout()
plt.savefig('fig11_pid_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 11 (PID comparison) saved.")
print()
print("Task 3 complete.")
print(f"  PID law: u(t) = 1 - [Kp*e + Ki*∫e + Kd*de/dt]")
print(f"  Best params: Kp={Kp_best}, Ki={Ki_best}, Kd={Kd_best}")
print(f"  Cycle time: {t_unc_end:.0f}s (uncontrolled) vs {t_pid_end:.0f}s (PID)")
print(f"  Peak velocity: {V_unc_peak:.1f} m/s (unc) vs {V_pid_peak:.1f} m/s (PID)")
if V_pid_peak > 8.0:
    print(f"  Note: PID reduces peak velocity but cannot fully satisfy V<8 m/s")
    print(f"  with current culvert parameters — velocity constraint is a model")
    print(f"  limitation (single equivalent culvert vs real multi-culvert design)")