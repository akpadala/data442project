"""
transfer_function.py
=====================
Analytical transfer function derivation for the Panama Canal FINAL_DRAIN phase.

System equations (FINAL_DRAIN phase):
--------------------------------------
Water hammer ODE:
    L * dQ/dt + R * Q = rho * g * delta_H(t)        ... (1)

where delta_H(t) = H_L(t) - H_sea(t)

Lock level ODE:
    dH_L/dt = -Q / A_L                               ... (2)

Tidal forcing:
    H_sea(t) = A_tide * sin(omega_tide * t)          ... (3)

Derivation
----------
Taking the Laplace transform of (1):
    L*s*Q(s) + R*Q(s) = rho*g * DeltaH(s)
    Q(s) / DeltaH(s) = (rho*g) / (L*s + R)          ... G1(s)

This is a first-order system with:
    DC gain  K  = rho*g / R
    Time constant tau = L / R
    Pole at  s = -R/L

Taking the Laplace transform of (2):
    s * H_L(s) - H_L(0) = -Q(s) / A_L
    H_L(s) = H_L(0)/s - Q(s) / (A_L * s)

Substituting Q(s) = G1(s) * DeltaH(s) and DeltaH(s) = H_L(s) - H_sea(s):
    H_L(s) = H_L(0)/s - G1(s)*(H_L(s) - H_sea(s)) / (A_L*s)

Solving for H_L(s):
    H_L(s) * [1 + G1(s)/(A_L*s)] = H_L(0)/s + G1(s)*H_sea(s)/(A_L*s)

    H_L(s) / H_sea(s) = G1(s) / (A_L*s + G1(s))    ... G2(s)  [zero-IC]

Substituting G1(s) = rho*g / (L*s + R):

    G2(s) = (rho*g / (L*s + R)) / (A_L*s + rho*g/(L*s+R))
           = rho*g / (A_L*s*(L*s + R) + rho*g)
           = (rho*g / (A_L*L)) / (s^2 + (R/L)*s + rho*g/(A_L*L))

This is a standard second-order system:
    omega_n^2 = rho*g / (A_L * L)
    2*zeta*omega_n = R / L
    zeta = R / (2 * L * omega_n)

Figures generated:
    fig7_bode.png         - Bode plot of G2(s)
    fig8_step_response.png - Step response of G2(s)
    fig9_tf_validation.png - Analytical steady-state vs simulation during FINAL_DRAIN
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp

import parameters as P
from dynamics import system_dynamics, H_sea, STATE_NAMES
from simulation import run_lock_cycle

# ── Numerical parameter values ────────────────────────────────────────────────
RHO    = P.RHO
G      = P.G
L_in   = P.INERTANCE          # hydraulic inertance  [kg/m^4]
R_in   = P.RESISTANCE         # hydraulic resistance [kg/m^7 ... see notes]
A_L    = P.LOCK_AREA           # lock surface area    [m^2]
A_tide = P.TIDE_AMPLITUDE      # tidal amplitude      [m]
w_tide = P.TIDE_OMEGA          # tidal frequency      [rad/s]

# ── Transfer function G1(s) = Q(s)/DeltaH(s) ─────────────────────────────────
# G1(s) = (rho*g) / (L*s + R)
# numerator: [rho*g],  denominator: [L, R]
K1     = RHO * G               # DC gain numerator
G1_num = [K1]
G1_den = [L_in, R_in]
G1     = signal.TransferFunction(G1_num, G1_den)

print("=" * 60)
print("TRANSFER FUNCTION DERIVATION — Panama Canal FINAL_DRAIN")
print("=" * 60)
print()
print("G1(s) = Q(s) / ΔH(s)")
print(f"      = {K1:.2f} / ({L_in:.4f}·s + {R_in:.6f})")
print(f"      = {K1/L_in:.2f} / (s + {R_in/L_in:.6f})")
print()
print(f"  Pole of G1:       s = -{R_in/L_in:.6f} rad/s")
print(f"  Time constant τ:  {L_in/R_in:.2f} s  ({L_in/R_in/60:.1f} min)")
print(f"  DC gain K:        {K1/R_in:.2f} m³/s per meter head")
print()

# ── Transfer function G2(s) = H_L(s)/H_sea(s) ────────────────────────────────
# G2(s) = (rho*g / (A_L*L)) / (s^2 + (R/L)*s + rho*g/(A_L*L))
wn2    = (RHO * G) / (A_L * L_in)      # omega_n^2
wn     = np.sqrt(wn2)                   # natural frequency [rad/s]
zeta   = R_in / (2 * L_in * wn)        # damping ratio
alpha  = R_in / L_in                    # 2*zeta*omega_n

G2_num = [wn2]
G2_den = [1.0, alpha, wn2]
G2     = signal.TransferFunction(G2_num, G2_den)

print("G2(s) = H_L(s) / H_sea(s)  [lock response to tidal forcing]")
print(f"      = {wn2:.6f} / (s² + {alpha:.6f}·s + {wn2:.6f})")
print()
print(f"  Natural frequency ωn:   {wn:.6f} rad/s  (period = {2*np.pi/wn/3600:.1f} hr)")
print(f"  Damping ratio ζ:        {zeta:.4f}")
if zeta < 1:
    print(f"  System regime:          UNDERDAMPED  (oscillatory transient)")
    wd = wn * np.sqrt(1 - zeta**2)
    print(f"  Damped frequency ωd:    {wd:.6f} rad/s")
elif zeta == 1:
    print(f"  System regime:          CRITICALLY DAMPED")
else:
    print(f"  System regime:          OVERDAMPED  (no oscillation)")
    print(f"  → Canal culverts are deliberately overdamped: prevents water hammer / mooring surges")
print()
print(f"  Poles of G2:")
poles = np.roots(G2_den)
for p in poles:
    print(f"    s = {p:.6f}")
print()

# Steady-state gain to sinusoidal input at tidal frequency
w_eval    = w_tide
_, H_mag, H_phase = signal.bode(G2, w=[w_eval])
ss_gain   = 10 ** (H_mag[0] / 20)   # convert dB to linear
ss_phase  = np.radians(H_phase[0])
print(f"Steady-state response to tidal forcing (ω = {w_tide:.2e} rad/s):")
print(f"  |G2(jω)|  = {ss_gain:.6f}  (lock amplitude / tidal amplitude)")
print(f"  ∠G2(jω)  = {np.degrees(ss_phase):.2f}°  (phase lag)")
print(f"  Lock amplitude at steady state ≈ {ss_gain * A_tide:.4f} m")
print()
print("Physical interpretation:")
print(f"  The lock level responds to the {A_tide}m tidal oscillation with an")
print(f"  amplitude of {ss_gain*A_tide:.4f}m — attenuated by {(1-ss_gain)*100:.1f}%.")
print(f"  The tidal forcing is at ω={w_tide:.2e} rad/s, far below ωn={wn:.2e} rad/s,")
print(f"  so the system is in the low-frequency regime and tracks the tide closely.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7: Bode plot of G2(s)
# ═══════════════════════════════════════════════════════════════════════════════
w_range = np.logspace(-7, -2, 500)   # rad/s — spans tidal to fast dynamics
w_out, mag_db, phase_deg = signal.bode(G2, w=w_range)

fig, (ax_mag, ax_phs) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle('Bode Plot — $G_2(s) = H_L(s)/H_{sea}(s)$\nLock Level Response to Tidal Forcing',
             fontsize=13, fontweight='bold')

# Magnitude
ax_mag.semilogx(w_out, mag_db, color='#2563EB', lw=2)
ax_mag.axvline(wn,     color='#DC2626', ls='--', lw=1.5, label=f'ωₙ = {wn:.2e} rad/s')
ax_mag.axvline(w_tide, color='#059669', ls=':',  lw=1.5, label=f'ω_tide = {w_tide:.2e} rad/s')
ax_mag.axhline(-3,     color='gray',   ls=':',  lw=1,   label='-3 dB')
ax_mag.set_ylabel('Magnitude (dB)', fontsize=10)
ax_mag.legend(fontsize=9, loc='lower left')
ax_mag.grid(True, which='both', alpha=0.3)
ax_mag.set_ylim(-60, 5)

# Phase
ax_phs.semilogx(w_out, phase_deg, color='#7C3AED', lw=2)
ax_phs.axvline(wn,     color='#DC2626', ls='--', lw=1.5)
ax_phs.axvline(w_tide, color='#059669', ls=':',  lw=1.5)
ax_phs.axhline(-90,    color='gray',   ls=':',  lw=1,   label='-90° reference')
ax_phs.set_ylabel('Phase (°)', fontsize=10)
ax_phs.set_xlabel('Frequency (rad/s)', fontsize=10)
ax_phs.legend(fontsize=9)
ax_phs.grid(True, which='both', alpha=0.3)

# Annotation box
ax_mag.text(0.02, 0.05,
    f'ωₙ = {wn:.2e} rad/s\nζ = {zeta:.4f}  ({"underdamped" if zeta<1 else "overdamped"})\n'
    f'τ = {L_in/R_in:.0f} s  ({L_in/R_in/60:.1f} min)',
    transform=ax_mag.transAxes, fontsize=8.5,
    bbox=dict(facecolor='white', edgecolor='#CBD5E1', boxstyle='round,pad=0.4'),
    va='bottom')

plt.tight_layout()
plt.savefig('fig7_bode.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 7 (Bode plot) saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8: Step response of G2(s)
# ═══════════════════════════════════════════════════════════════════════════════
t_step = np.linspace(0, min(5 * L_in/R_in, 27000), 2000)  # 5τ or 450 min — full decay envelope visible
t_out, y_out = signal.step(G2, T=t_step)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t_out / 60, y_out, color='#2563EB', lw=2, label='Step response H_L(t)')
ax.axhline(1.0, color='gray', ls=':', lw=1, label='Final value (1.0)')
ax.axhline(0.632, color='#DC2626', ls='--', lw=1, label='63.2% (one τ)')

# Mark time constant
tau = L_in / R_in
ax.axvline(tau / 60, color='#DC2626', ls='--', lw=1)
ax.text(tau/60 + 0.05, 0.1, f'τ = {tau:.0f}s\n({tau/60:.1f} min)',
        fontsize=8.5, color='#DC2626')

if zeta < 1:
    # Mark overshoot
    idx_peak = np.argmax(y_out)
    os_pct   = (y_out[idx_peak] - 1.0) * 100
    ax.annotate(f'Overshoot\n{os_pct:.1f}%',
                xy=(t_out[idx_peak]/60, y_out[idx_peak]),
                xytext=(t_out[idx_peak]/60 + 1, y_out[idx_peak] + 0.05),
                arrowprops=dict(arrowstyle='->', color='#059669'),
                fontsize=8.5, color='#059669')

ax.set_xlabel('Time (minutes)', fontsize=10)
ax.set_ylabel('Normalised H_L response', fontsize=10)
ax.set_title('Step Response — $G_2(s) = H_L(s)/H_{sea}(s)$\n'
             f'ζ = {zeta:.4f}  |  ωₙ = {wn:.2e} rad/s  |  τ = {tau:.0f} s',
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)

# Annotate damping regime
regime = 'Overdamped (ζ > 1)\nNo oscillation — canal is deliberately overdamped\nto prevent mooring surges (water hammer protection)' \
    if zeta >= 1 else f'Underdamped (ζ = {zeta:.3f})\nOscillatory transient'
ax.text(0.98, 0.05, regime, transform=ax.transAxes, fontsize=8.5,
        ha='right', va='bottom',
        bbox=dict(facecolor='#FEF9C3', edgecolor='#D97706',
                  boxstyle='round,pad=0.4'))

plt.tight_layout()
plt.savefig('fig8_step_response.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 8 (Step response) saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9: Analytical steady-state vs simulation — FINAL_DRAIN validation
# ═══════════════════════════════════════════════════════════════════════════════

# Run full simulation to get FINAL_DRAIN segment
t_sim, x_sim, log = run_lock_cycle()

# Extract FINAL_DRAIN segment
fd_name = 'Final drain (lock→sea)'
fd_entry = next((entry for entry in log if entry[0] == fd_name), None)

if fd_entry:
    t_fd_start = fd_entry[1]
    t_fd_end   = fd_entry[2]
    mask       = (t_sim >= t_fd_start) & (t_sim <= t_fd_end)
    t_fd       = t_sim[mask] - t_fd_start   # time relative to phase entry
    hL_fd      = x_sim[0, mask]
    Q_fd       = x_sim[4, mask]

    # Analytical steady-state: H_L_ss(t) = H_L_initial + ss_gain * A_tide * sin(w_tide*t + ss_phase + phi)
    # More precisely: solve the full forced response
    # Initial condition: H_L(0) = hL_fd[0], Q(0) = 0
    # System: s^2 Y + alpha*s*Y + wn^2*Y = wn^2 * H_sea(t)
    # Numerical solution using scipy for comparison
    def ode_second_order(t, y):
        # y = [H_L, dH_L/dt]
        # d2H_L/dt2 = -alpha * dH_L/dt - wn^2*(H_L - H_sea(t+t_fd_start))
        H_sea_t = A_tide * np.sin(w_tide * (t + t_fd_start) + P.TIDE_PHASE)
        d2H = -alpha * y[1] - wn2 * (y[0] - H_sea_t)
        return [y[1], d2H]

    y0_tf  = [hL_fd[0], 0.0]
    t_eval_uniform = np.linspace(0, t_fd[-1], len(t_fd))
    sol_tf = solve_ivp(ode_second_order,
                       [0, t_fd[-1]],
                       y0_tf,
                       method='RK45', max_step=1.0,
                       t_eval=t_eval_uniform)

    hL_analytical = sol_tf.y[0]
    t_anal        = sol_tf.t   # uniform grid

    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle('Transfer Function Validation — FINAL_DRAIN Phase\n'
                 'Analytical G₂(s) vs ODE Simulation',
                 fontsize=13, fontweight='bold')

    ax_h, ax_q = axes

    # H_L comparison
    ax_h.plot(t_fd / 60, hL_fd,              color='#2563EB', lw=2,   label='ODE simulation H_L(t)')
    ax_h.plot(t_anal / 60, hL_analytical,    color='#DC2626', lw=1.5, ls='--',
              label='Transfer function G₂(s) response')
    t_tide_fine = np.linspace(0, t_fd[-1], 1000)
    ax_h.plot(t_tide_fine / 60,
              A_tide * np.sin(w_tide * (t_tide_fine + t_fd_start) + P.TIDE_PHASE),
              color='#0891B2', lw=1, ls=':', alpha=0.7, label='H_sea(t) tidal input')
    ax_h.set_ylabel('Water level (m)', fontsize=10)
    ax_h.legend(fontsize=9)
    ax_h.grid(True, alpha=0.3)

    # Residual
    # Interpolate analytical solution onto simulation time grid for residual
    hL_anal_interp = np.interp(t_fd, t_anal, hL_analytical)
    residual = hL_anal_interp - hL_fd
    rms_err  = np.sqrt(np.mean(residual**2))
    ax_q.plot(t_fd / 60, residual,
              color='#7C3AED', lw=1.5, label=f'Residual (TF − simulation)')
    ax_q.axhline(0, color='gray', lw=0.8, ls=':')
    ax_q.set_ylabel('Residual H_L (m)', fontsize=10)
    ax_q.set_xlabel('Time within FINAL_DRAIN phase (minutes)', fontsize=10)
    ax_q.legend(fontsize=9)
    ax_q.grid(True, alpha=0.3)

    # RMS error annotation
    ax_h.text(0.98, 0.05,
              f'RMS error: {rms_err:.4f} m\n({rms_err/hL_fd[0]*100:.2f}% of initial H_L)',
              transform=ax_h.transAxes, fontsize=9, ha='right', va='bottom',
              bbox=dict(facecolor='white', edgecolor='#059669', boxstyle='round,pad=0.4'))

    plt.tight_layout()
    plt.savefig('fig9_tf_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Fig 9 (TF validation) saved.  RMS error = {rms_err:.4f} m")
else:
    print("Warning: FINAL_DRAIN segment not found in simulation log.")

print()
print("Task 1 complete.")
print(f"  G1(s): pole at s = -{R_in/L_in:.4f}, τ = {L_in/R_in:.0f}s")
print(f"  G2(s): ωn = {wn:.4e} rad/s, ζ = {zeta:.4f}, regime = {'overdamped' if zeta>=1 else 'underdamped'}")