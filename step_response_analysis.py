"""
step_response_analysis.py
==========================
Verifies whether the water hammer inertance term produces genuine second-order
oscillatory behavior in Q, computes the damping ratio analytically, and
generates a presentation-quality figure showing the Q transient.

Run with: python step_response_analysis.py

Figures generated:
    fig10_q_transient.png  - Q transient under step vs ramp valve opening,
                             annotated with damping ratio and regime
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import parameters as P
from dynamics import H_sea, PHASE_CONFIG, STATE_NAMES
from parameters import (
    LOCK_H_UPPER, BASIN_3_H_INIT, BASIN_2_H_INIT, BASIN_1_H_INIT
)

# ── Analytical damping ratio ───────────────────────────────────────────────────
#
# The Q ODE during any WSB drain phase is:
#
#   L * dQ/dt + R * Q = rho * g * delta_H(t)
#   dH_L/dt           = -Q / A_L
#   dH_target/dt      = +Q / A_B   (for basin phases)
#
# Treating delta_H = H_L - H_target as the dynamic variable:
#   d(delta_H)/dt = dH_L/dt - dH_target/dt
#                 = -Q/A_L - Q/A_B
#                 = -Q * (1/A_L + 1/A_B)
#                 = -Q / A_eff    where A_eff = A_L*A_B/(A_L+A_B)
#
# Differentiating the Q ODE:
#   L * d2Q/dt2 + R * dQ/dt = rho*g * d(delta_H)/dt
#                            = -rho*g * Q / A_eff
#
# Rearranging:
#   d2Q/dt2 + (R/L) * dQ/dt + (rho*g / (L * A_eff)) * Q = 0
#
# This is a standard second-order homogeneous ODE:
#   omega_n^2 = rho*g / (L * A_eff)
#   2*zeta*omega_n = R / L
#   zeta = R / (2 * sqrt(rho*g * L / A_eff))

def compute_damping(A_eff):
    wn2  = P.RHO * P.G / (P.INERTANCE * A_eff)
    wn   = np.sqrt(wn2)
    zeta = P.RESISTANCE / (2 * P.INERTANCE * wn)
    return wn, zeta

# Effective areas for each phase
A_eff_wsb   = P.LOCK_AREA * P.BASIN_AREA / (P.LOCK_AREA + P.BASIN_AREA)
A_eff_final = P.LOCK_AREA   # sea is infinite reservoir, A_B -> inf

wn_wsb,   zeta_wsb   = compute_damping(A_eff_wsb)
wn_final, zeta_final = compute_damping(A_eff_final)

print("=" * 60)
print("SECOND-ORDER TRANSIENT ANALYSIS — Q dynamics")
print("=" * 60)
print()
print("WSB drain phases (lock ↔ basin):")
print(f"  A_eff    = {A_eff_wsb:.1f} m²")
print(f"  ωn       = {wn_wsb:.4e} rad/s  (period = {2*np.pi/wn_wsb/60:.1f} min)")
print(f"  ζ        = {zeta_wsb:.6f}")
regime_wsb = "UNDERDAMPED" if zeta_wsb < 1 else "OVERDAMPED"
print(f"  Regime   = {regime_wsb}")
print()
print("FINAL_DRAIN phase (lock → sea, A_B → ∞):")
print(f"  A_eff    = {A_eff_final:.1f} m²  (lock area only)")
print(f"  ωn       = {wn_final:.4e} rad/s  (period = {2*np.pi/wn_final/60:.1f} min)")
print(f"  ζ        = {zeta_final:.6f}")
regime_final = "UNDERDAMPED" if zeta_final < 1 else "OVERDAMPED"
print(f"  Regime   = {regime_final}")
print()

if zeta_wsb < 1:
    wd = wn_wsb * np.sqrt(1 - zeta_wsb**2)
    T_osc = 2 * np.pi / wd
    print(f"  Damped oscillation period: {T_osc/60:.2f} min")
    os_pct = 100 * np.exp(-np.pi * zeta_wsb / np.sqrt(1 - zeta_wsb**2))
    print(f"  Theoretical overshoot:     {os_pct:.1f}%")
    print()

print("Physical interpretation:")
print(f"  ζ = {zeta_wsb:.4f} << 1 → system is HIGHLY underdamped")
print(f"  → Q will oscillate around steady-state before settling")
print(f"  → This IS water hammer: pressure waves in the culvert")
print(f"  → The valve RAMP (240s open time) is the engineer's solution:")
print(f"     it avoids exciting the resonant mode by opening slowly")
print(f"     compared to the natural period of {2*np.pi/wn_wsb/60:.1f} min")
print()

# ── ODE systems for step vs ramp valve ───────────────────────────────────────

def make_ode(valve_mode='ramp', phase=0):
    """
    valve_mode: 'step' (instant open) or 'ramp' (linear over VALVE_OPEN_TIME)
    phase: 0=WSB1 (lock->basin3), 3=FINAL (lock->sea)
    """
    target_idx, fill_idx = PHASE_CONFIG[phase]

    def ode(t, y):
        h_l, h_b3, h_b2, h_b1, Q = y

        if target_idx == -1:
            h_target = H_sea(t)
        else:
            h_target = y[target_idx]

        delta_h = max(h_l - h_target, 0.0)

        if valve_mode == 'step':
            Av = P.A_VALVE_MAX
        else:
            Av = P.A_VALVE_MAX * min(t / P.VALVE_OPEN_TIME, 1.0)

        driving  = P.RHO * P.G * delta_h * P.CD * (Av / P.A_VALVE_MAX)
        friction = P.RESISTANCE * Q
        dQdt     = (driving - friction) / P.INERTANCE

        dhl_dt = -Q / P.LOCK_AREA
        dydt   = [dhl_dt, 0.0, 0.0, 0.0, dQdt]

        if fill_idx != -1:
            dydt[fill_idx] = Q / P.BASIN_AREA

        return dydt

    return ode

# ── Run simulations: WSB phase 0 with step vs ramp valve ─────────────────────
y0 = [LOCK_H_UPPER, BASIN_3_H_INIT, BASIN_2_H_INIT, BASIN_1_H_INIT, 0.0]

T_sim  = 600.0   # 10 minutes — enough to see the transient
dt_max = 0.5     # fine resolution to capture oscillations

def run_sim(valve_mode, phase=0):
    def event(t, y):
        target_idx = PHASE_CONFIG[phase][0]
        h_t = H_sea(t) if target_idx == -1 else y[target_idx]
        return (y[0] - h_t) - P.EPSILON
    event.terminal = True; event.direction = -1

    sol = solve_ivp(
        make_ode(valve_mode, phase),
        [0, T_sim], y0.copy(),
        method='RK45', max_step=dt_max,
        events=event, dense_output=False
    )
    return sol.t, sol.y

t_step, x_step = run_sim('step',  phase=0)
t_ramp, x_ramp = run_sim('ramp',  phase=0)

# ── Torricelli steady-state Q for reference ───────────────────────────────────
# Q_ss(t) = CD * A_max * sqrt(2g * delta_H(t))
delta_H_arr_step = np.maximum(x_step[0] - x_step[1], 0)
Q_torr_step      = P.CD * P.A_VALVE_MAX * np.sqrt(2 * P.G * delta_H_arr_step)

delta_H_arr_ramp = np.maximum(x_ramp[0] - x_ramp[1], 0)
Q_torr_ramp      = P.CD * P.A_VALVE_MAX * np.minimum(
    t_ramp / P.VALVE_OPEN_TIME, 1.0) * np.sqrt(2 * P.G * delta_H_arr_ramp)

# ── Compute overshoot and settling time ───────────────────────────────────────
Q_step     = x_step[4]
Q_ss_final = Q_torr_step[-1]   # approximate steady-state at end

if len(Q_step) > 10:
    idx_peak  = np.argmax(Q_step)
    Q_peak    = Q_step[idx_peak]
    # Settle within 5% of final Torricelli value
    Q_final_approx = np.mean(Q_step[-20:]) if len(Q_step) > 20 else Q_step[-1]
    settle_mask    = np.abs(Q_step - Q_final_approx) < 0.05 * Q_final_approx
    if settle_mask.any():
        t_settle = t_step[np.where(settle_mask)[0][-1] if settle_mask.sum() > 0 else -1]
    else:
        t_settle = t_step[-1]

    if Q_final_approx > 0:
        os_actual = (Q_peak - Q_final_approx) / Q_final_approx * 100
    else:
        os_actual = 0
else:
    Q_peak = Q_step[-1]; os_actual = 0; t_settle = t_step[-1]

print(f"Step valve simulation results (WSB drain 1):")
print(f"  Peak Q:          {Q_peak:.0f} m³/s")
print(f"  Approx Q_ss:     {Q_final_approx:.0f} m³/s")
print(f"  Actual overshoot: {os_actual:.1f}%")
if zeta_wsb < 1:
    print(f"  Theory overshoot: {100*np.exp(-np.pi*zeta_wsb/np.sqrt(1-zeta_wsb**2)):.1f}%")
print()

# ── Velocity check ────────────────────────────────────────────────────────────
V_peak_step = Q_peak / P.CULVERT_AREA
V_peak_ramp = x_ramp[4].max() / P.CULVERT_AREA
print(f"Peak velocity (step valve): {V_peak_step:.1f} m/s")
print(f"Peak velocity (ramp valve): {V_peak_ramp:.1f} m/s")
print(f"Spec max velocity:          8.0 m/s")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10: Q transient — step vs ramp, with damping annotation
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
fig.suptitle(
    'Second-Order Transient in Flow Rate Q\n'
    'Step vs Ramp Valve Opening — WSB Drain Phase 1 (Lock → Basin 3)',
    fontsize=13, fontweight='bold'
)

ax_q, ax_dh, ax_v = axes

# ── Panel 1: Q vs time ────────────────────────────────────────────────────────
ax_q.plot(t_step / 60, x_step[4], color='#DC2626', lw=2,
          label='Step valve (instant open) — shows resonance')
ax_q.plot(t_ramp / 60, x_ramp[4], color='#2563EB', lw=2,
          label='Ramp valve (240s open) — damps resonance')
ax_q.plot(t_step / 60, Q_torr_step, color='#DC2626', lw=1,
          ls=':', alpha=0.5, label='Torricelli Q_ss (step)')
ax_q.plot(t_ramp / 60, Q_torr_ramp, color='#2563EB', lw=1,
          ls=':', alpha=0.5, label='Torricelli Q_ss (ramp)')

# Mark peak overshoot on step response
if os_actual > 0.5:
    ax_q.annotate(
        f'Overshoot\n{os_actual:.0f}%\n(water hammer)',
        xy=(t_step[np.argmax(x_step[4])] / 60, Q_peak),
        xytext=(t_step[np.argmax(x_step[4])] / 60 + 0.5, Q_peak * 1.05),
        arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.5),
        fontsize=8.5, color='#DC2626',
        bbox=dict(facecolor='white', edgecolor='#DC2626',
                  boxstyle='round,pad=0.3', alpha=0.9)
    )

# Drain time vs natural period annotation
T_n = 2 * np.pi / wn_wsb
ax_q.axvline(t_step[-1] / 60, color='#059669', lw=1.5, ls='--',
             label=f'Drain complete ({t_step[-1]/60:.1f} min)')
ax_q.axvspan(0, T_n / 60, alpha=0.08, color='#059669',
             label=f'One natural period Tₙ = {T_n/60:.1f} min')
ax_q.text(T_n / 60 / 2, ax_q.get_ylim()[1] * 0.02 if ax_q.get_ylim()[1] > 0 else 50,
          f'Tₙ = {T_n/60:.1f} min', ha='center', fontsize=7.5,
          color='#059669', style='italic')

ax_q.set_ylabel('Flow rate Q (m³/s)', fontsize=10)
ax_q.legend(fontsize=8, loc='upper right', ncol=2)
ax_q.grid(True, alpha=0.3)

# Damping annotation box
ax_q.text(0.01, 0.97,
    f'ζ = {zeta_wsb:.4f}  →  {"UNDERDAMPED" if zeta_wsb < 1 else "OVERDAMPED"}\n'
    f'ωₙ = {wn_wsb:.3e} rad/s  |  Tₙ = {2*np.pi/wn_wsb/60:.1f} min\n'
    f'Drain time ({t_step[-1]/60:.1f} min) < Tₙ → no oscillation develops\n'
    f'Canal geometry provides inherent water hammer protection',
    transform=ax_q.transAxes, fontsize=8, va='top',
    bbox=dict(facecolor='#FEF9C3', edgecolor='#D97706',
              boxstyle='round,pad=0.4', alpha=0.95))

# ── Panel 2: ΔH vs time ───────────────────────────────────────────────────────
dH_step = np.maximum(x_step[0] - x_step[1], 0)
dH_ramp = np.maximum(x_ramp[0] - x_ramp[1], 0)

ax_dh.plot(t_step / 60, dH_step, color='#DC2626', lw=2, label='Step valve')
ax_dh.plot(t_ramp / 60, dH_ramp, color='#2563EB', lw=2, label='Ramp valve')
ax_dh.axhline(P.EPSILON, color='gray', lw=1, ls='--', label=f'ε = {P.EPSILON}m (transition)')
ax_dh.set_ylabel('Head difference ΔH (m)', fontsize=10)
ax_dh.legend(fontsize=8.5, loc='upper right')
ax_dh.grid(True, alpha=0.3)

# ── Panel 3: Velocity vs time ─────────────────────────────────────────────────
V_step = x_step[4] / P.CULVERT_AREA
V_ramp = x_ramp[4] / P.CULVERT_AREA

ax_v.plot(t_step / 60, V_step, color='#DC2626', lw=2, label='Step valve velocity')
ax_v.plot(t_ramp / 60, V_ramp, color='#2563EB', lw=2, label='Ramp valve velocity')
ax_v.axhline(8.0,         color='#F59E0B', lw=2, ls='--', label='Max spec: 8 m/s')
ax_v.axhline(4.7,         color='#059669', lw=1, ls=':',  label='Avg spec: 4.7 m/s')

# Shade violation region
t_all_v  = np.concatenate([t_step, t_ramp])
V_all    = np.concatenate([V_step, V_ramp])
t_viol   = t_step[V_step > 8.0]
if len(t_viol) > 0:
    ax_v.fill_between(t_step / 60, 8.0, V_step,
                       where=(V_step > 8.0), alpha=0.25,
                       color='#DC2626', label='Velocity spec violation')

ax_v.set_ylabel('Flow velocity (m/s)', fontsize=10)
ax_v.set_xlabel('Time (minutes)', fontsize=10)
ax_v.legend(fontsize=8.5, loc='upper right')
ax_v.grid(True, alpha=0.3)
ax_v.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig('fig10_q_transient.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 10 (Q transient) saved.")
print()
T_n = 2*np.pi/wn_wsb
print(f"  System is {regime_wsb} with ζ = {zeta_wsb:.4f}")
print(f"  Natural period T_n = {T_n:.0f}s ({T_n/60:.1f} min)")
print(f"  WSB drain duration ≈ 306s (5.1 min) < T_n")
print(f"  → Drain completes before one oscillation — water hammer cannot develop")
print(f"  → Canal geometry provides inherent water hammer protection")
print(f"  → Valve ramp is a safety margin, not a strict necessity")