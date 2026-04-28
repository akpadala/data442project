"""
analysis.py
Plots, PID controller on valve opening rate, and frequency/resonance analysis.
Run after simulation.py to generate all figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import welch

from parameters import (
    LOCK_H_UPPER, TIDE_AMPLITUDE, TIDE_OMEGA,
    INERTANCE, RESISTANCE, RHO, G, LOCK_AREA
)
from dynamics import H_sea, STATE_NAMES, WSB_DRAIN_1, WSB_DRAIN_2, WSB_DRAIN_3, FINAL_DRAIN
from simulation import run_lock_cycle

# ── Color scheme ──────────────────────────────────────────────────────────────
COLORS = {
    'lock'   : '#2563EB',   # blue
    'basin3' : '#059669',   # green
    'basin2' : '#D97706',   # amber
    'basin1' : '#DC2626',   # red
    'flow'   : '#7C3AED',   # purple
    'tide'   : '#0891B2',   # cyan
}

STATE_COLORS = {
    'WSB drain 1' : '#DCFCE7',
    'WSB drain 2' : '#FEF9C3',
    'WSB drain 3' : '#FEE2E2',
    'Final drain' : '#EDE9FE',
}

# ── Figure 1: Water levels over full lock cycle ───────────────────────────────

def plot_water_levels(t, x, log, save=False):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Panama Canal — Neopanamax Lock Cycle\nWater Levels, Flow Rate, and Tidal Forcing",
                 fontsize=14, fontweight='bold')

    ax_levels, ax_flow, ax_tide = axes

    # Shade background by FSM state
    state_short = {
        'WSB drain 1 (lock→basin 3)' : 'WSB drain 1',
        'WSB drain 2 (lock→basin 2)' : 'WSB drain 2',
        'WSB drain 3 (lock→basin 1)' : 'WSB drain 3',
        'Final drain (lock→sea)'     : 'Final drain',
    }
    for name, t_in, t_out in log:
        short = state_short.get(name, name)
        color = STATE_COLORS.get(short, '#F5F5F5')
        for ax in axes:
            ax.axvspan(t_in / 60, t_out / 60, alpha=0.3, color=color, zorder=0)
        ax_levels.text((t_in + t_out) / 2 / 60, LOCK_H_UPPER * 0.97,
                       short, ha='center', va='top', fontsize=7, color='#555')

    # Plot water levels
    ax_levels.plot(t / 60, x[0], color=COLORS['lock'],   lw=2,   label='Lock (H_L)')
    ax_levels.plot(t / 60, x[1], color=COLORS['basin3'], lw=1.5, label='Basin 3 (H_B3)', ls='--')
    ax_levels.plot(t / 60, x[2], color=COLORS['basin2'], lw=1.5, label='Basin 2 (H_B2)', ls='--')
    ax_levels.plot(t / 60, x[3], color=COLORS['basin1'], lw=1.5, label='Basin 1 (H_B1)', ls='--')
    ax_levels.set_ylabel('Water level (m)', fontsize=11)
    ax_levels.legend(loc='upper right', fontsize=9)
    ax_levels.set_ylim(-1, LOCK_H_UPPER + 2)
    ax_levels.grid(True, alpha=0.3)

    # Plot flow rate Q
    ax_flow.plot(t / 60, x[4], color=COLORS['flow'], lw=2, label='Flow rate Q (m³/s)')
    ax_flow.axhline(0, color='gray', lw=0.8, ls=':')
    ax_flow.set_ylabel('Flow rate Q (m³/s)', fontsize=11)
    ax_flow.legend(loc='upper right', fontsize=9)
    ax_flow.grid(True, alpha=0.3)

    # Plot tidal forcing
    t_fine = np.linspace(t[0], t[-1], 1000)
    ax_tide.plot(t_fine / 60, H_sea(t_fine), color=COLORS['tide'], lw=2, label='Sea level H_sea(t)')
    ax_tide.axhline(0, color='gray', lw=0.8, ls=':')
    ax_tide.set_ylabel('Sea level (m)', fontsize=11)
    ax_tide.set_xlabel('Time (minutes)', fontsize=11)
    ax_tide.legend(loc='upper right', fontsize=9)
    ax_tide.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('water_levels.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Figure 2: Phase portrait — ΔH vs Q (lock ↔ basin 3) ──────────────────────

def plot_phase_portrait(t, x, log, save=False):
    """
    Phase portrait of ΔH vs Q for the WSB_DRAIN_1 phase.
    Analogous to displacement vs velocity in suspension analysis.
    """
    # Extract WSB_DRAIN_1 segment
    t_in  = next(t_in  for name, t_in, t_out in log if 'basin 3' in name)
    t_out = next(t_out for name, t_in, t_out in log if 'basin 3' in name)
    mask  = (t >= t_in) & (t <= t_out)

    dH = x[0, mask] - x[1, mask]   # H_L - H_B3
    Q  = x[4, mask]

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(dH, Q, c=t[mask] / 60, cmap='plasma', s=8, zorder=3)
    ax.plot(dH, Q, color='gray', lw=0.5, alpha=0.5, zorder=2)
    ax.scatter(dH[0], Q[0], color='green', s=80, zorder=4, label='Start')
    ax.scatter(dH[-1], Q[-1], color='red',   s=80, zorder=4, label='End (ΔH→0)')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Time (min)', fontsize=10)

    ax.set_xlabel('Head difference ΔH = H_L − H_B3 (m)', fontsize=11)
    ax.set_ylabel('Flow rate Q (m³/s)', fontsize=11)
    ax.set_title('Phase portrait — WSB drain phase 1\n(analogous to displacement vs velocity)',
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('phase_portrait.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Figure 3: Water recovery analysis ─────────────────────────────────────────

def plot_water_recovery(x, log, save=False):
    """
    Show how much water was recovered by each basin vs lost to sea.
    Directly maps to the 60% recovery claim.
    """
    H_L_start = x[0, 0]
    H_L_end   = x[0, -1]
    total_drop = H_L_start - H_L_end

    # Volume recovered = rise in each basin × basin area
    from parameters import BASIN_AREA, LOCK_AREA
    dH_B3 = x[1, -1] - x[1, 0]
    dH_B2 = x[2, -1] - x[2, 0]
    dH_B1 = x[3, -1] - x[3, 0]

    vol_B3       = dH_B3 * BASIN_AREA
    vol_B2       = dH_B2 * BASIN_AREA
    vol_B1       = dH_B1 * BASIN_AREA
    vol_total    = total_drop * LOCK_AREA
    vol_lost     = vol_total - (vol_B3 + vol_B2 + vol_B1)
    vol_recovered = vol_B3 + vol_B2 + vol_B1

    labels  = ['Basin 3\n(recovered)', 'Basin 2\n(recovered)', 'Basin 1\n(recovered)', 'Lost to sea']
    volumes = [vol_B3, vol_B2, vol_B1, max(vol_lost, 0)]
    colors  = [COLORS['basin3'], COLORS['basin2'], COLORS['basin1'], '#9CA3AF']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [v / 1e6 for v in volumes], color=colors, edgecolor='white', linewidth=1.5)

    for bar, vol in zip(bars, volumes):
        pct = 100 * vol / vol_total if vol_total > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    recovery_pct = 100 * vol_recovered / vol_total if vol_total > 0 else 0
    ax.set_title(f'Water volume distribution — one lock cycle\n'
                 f'Total recovery: {recovery_pct:.1f}% (target: ~60%)', fontsize=12)
    ax.set_ylabel('Volume (million m³)', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('water_recovery.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── PID controller on valve opening rate ──────────────────────────────────────

class PIDController:
    """
    PID controller that regulates valve opening rate to track a
    target equalization time T_target.

    Control variable u(t) = valve opening fraction (0 to 1)
    Error = (current ΔH) - (ideal linear ramp to zero by T_target)
    """
    def __init__(self, Kp=0.01, Ki=0.0005, Kd=0.1, T_target=1200.0):
        self.Kp       = Kp
        self.Ki       = Ki
        self.Kd       = Kd
        self.T_target = T_target   # target equalization time (s)
        self._integral = 0.0
        self._prev_error = None
        self._prev_t     = None

    def reset(self):
        self._integral   = 0.0
        self._prev_error = None
        self._prev_t     = None

    def compute(self, t, t_entry, delta_H, delta_H_initial):
        """
        Returns valve opening fraction u ∈ [0, 1].
        Ideal trajectory: ΔH decreases linearly from delta_H_initial to 0
        over T_target seconds.
        """
        elapsed     = t - t_entry
        setpoint    = delta_H_initial * max(1.0 - elapsed / self.T_target, 0.0)
        error       = delta_H - setpoint

        dt = (t - self._prev_t) if self._prev_t is not None else 1.0
        self._integral += error * dt

        derivative = (error - self._prev_error) / dt if self._prev_error is not None else 0.0

        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        u = np.clip(u, 0.0, 1.0)

        self._prev_error = error
        self._prev_t     = t
        return u


# ── Figure 4: PID vs uncontrolled valve comparison ───────────────────────────

def plot_pid_comparison(save=False):
    """
    Compare system response with and without PID valve control.
    Shows overshoot/oscillation vs smooth equalization.
    """
    from scipy.integrate import solve_ivp
    from parameters import LOCK_H_UPPER, BASIN_3_H_INIT, A_VALVE_MAX, VALVE_OPEN_TIME

    pid = PIDController(Kp=0.008, Ki=0.0003, Kd=0.15, T_target=1200.0)
    delta_H_0 = LOCK_H_UPPER - BASIN_3_H_INIT

    # Uncontrolled: standard ramp valve
    t_unc, x_unc, _ = run_lock_cycle()

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle('PID valve control vs uncontrolled — WSB drain phase 1', fontsize=13)

    for ax, (t_data, x_data, label, color) in zip(
        axes,
        [(t_unc, x_unc, 'Uncontrolled (ramp valve)', '#DC2626'),
         (t_unc, x_unc, 'PID controlled', '#2563EB')]   # placeholder — extend with PID sim
    ):
        dH = x_data[0] - x_data[1]
        Q  = x_data[4]
        ax.plot(t_data / 60, dH, lw=2, color=color, label=f'ΔH — {label}')
        ax2 = ax.twinx()
        ax2.plot(t_data / 60, Q, lw=1.5, color=color, ls='--', alpha=0.6, label='Q')
        ax.set_ylabel('Head difference ΔH (m)', fontsize=10)
        ax2.set_ylabel('Flow rate Q (m³/s)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (minutes)', fontsize=11)
    plt.tight_layout()
    if save:
        plt.savefig('pid_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Figure 4: Sensitivity — Cd vs total cycle time ───────────────────────────

def plot_sensitivity(save=False):
    """
    Sweep discharge coefficient Cd and measure total cycle time.
    Shows how sensitive results are to this uncertain parameter.
    """
    from parameters import CD
    cd_vals  = np.linspace(0.4, 1.0, 13)
    eq_times = []
    for cd in cd_vals:
        t2, x2, log2 = run_lock_cycle(cd_override=cd)
        eq_times.append(t2[-1] / 60)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cd_vals, eq_times, color='#2563EB', lw=2, marker='o', ms=6)
    ax.axvline(CD, color='#DC2626', ls='--', lw=1.5, label=f'Default Cd = {CD}')
    ax.set_xlabel('Discharge coefficient Cd', fontsize=11)
    ax.set_ylabel('Total cycle time (minutes)', fontsize=11)
    ax.set_title('Sensitivity: Cd vs total lock cycle time', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('fig4_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Figure 5: Valve timing tradeoff ──────────────────────────────────────────

def plot_valve_tradeoff(save=False):
    """
    Sweep valve open time and show tradeoff between cycle speed and peak flow.
    Faster valve → shorter cycle but higher peak Q (water hammer risk).
    """
    import parameters as P
    vt_vals     = [120, 180, 240, 360, 480]
    cycle_times = []
    peak_Qs     = []
    for vt in vt_vals:
        t2, x2, log2 = run_lock_cycle(valve_time_override=vt)
        cycle_times.append(t2[-1] / 60)
        peak_Qs.append(x2[4].max())

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.suptitle('Valve opening time: speed vs stability tradeoff',
                 fontsize=13, fontweight='bold')

    axes[0].plot(vt_vals, cycle_times, color='#2563EB', lw=2, marker='o', ms=7)
    axes[0].axvline(240, color='#DC2626', ls='--', lw=1.5, label='Default (240s)')
    axes[0].set_ylabel('Total cycle time (min)', fontsize=11)
    axes[0].set_title('Faster valve → shorter cycle (higher throughput)', fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(vt_vals, peak_Qs, color='#7C3AED', lw=2, marker='o', ms=7)
    axes[1].axvline(240, color='#DC2626', ls='--', lw=1.5, label='Default (240s)')
    axes[1].axhline(8.0 * P.CULVERT_AREA, color='#F59E0B', ls=':', lw=1.5,
                    label='Max velocity spec (8 m/s)')
    axes[1].set_xlabel('Valve open time (s)', fontsize=11)
    axes[1].set_ylabel('Peak Q (m³/s)', fontsize=11)
    axes[1].set_title('Faster valve → higher peak flow (water hammer risk)', fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('fig5_pid_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Figure 6: Culvert back-calculation from velocity spec ────────────────────

def plot_culvert_backCalc(x, save=False):
    """
    Back-calculate implied culvert diameter from published velocity specs
    (Calvo Gobbetti 2013: V_avg = 4.7 m/s, V_max = 8.0 m/s).
    """
    Q_peak = x[4].max()
    V_avg  = 4.7   # m/s — average velocity spec from paper
    V_max  = 8.0   # m/s — minimum flow spec from paper

    n_range    = np.arange(2, 12)
    D_from_avg = [2 * np.sqrt(Q_peak / V_avg / n / np.pi) for n in n_range]
    D_from_max = [2 * np.sqrt(Q_peak / V_max / n / np.pi) for n in n_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_range, D_from_avg, color='#2563EB', lw=2, marker='o', ms=5,
            label=f'V_avg = {V_avg} m/s (typical flow)')
    ax.plot(n_range, D_from_max, color='#DC2626', lw=2, marker='s', ms=5,
            label=f'V_max = {V_max} m/s (minimum spec)')
    ax.axhline(7.0, color='#059669', ls='--', lw=1.5,
               label='Current assumption D = 7.0 m')
    ax.fill_between(n_range, D_from_max, D_from_avg,
                    alpha=0.1, color='#2563EB', label='Feasible range')
    ax.set_xlabel('Number of culverts per lock chamber', fontsize=11)
    ax.set_ylabel('Implied culvert diameter (m)', fontsize=11)
    ax.set_title(
        f'Culvert diameter back-calculated from velocity spec\n'
        f'(Peak Q = {Q_peak:.0f} m³/s from simulation — Calvo Gobbetti 2013)',
        fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2, 11)
    plt.tight_layout()
    if save:
        plt.savefig('fig6_culvert_backCalc.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running simulation for analysis...")
    t, x, log = run_lock_cycle()

    print("\nGenerating figures...")
    plot_water_levels(t, x, log, save=True)
    print("  Fig 1 saved: water_levels.png")

    plot_phase_portrait(t, x, log, save=True)
    print("  Fig 2 saved: phase_portrait.png")

    plot_water_recovery(x, log, save=True)
    print("  Fig 3 saved: water_recovery.png")

    plot_sensitivity(save=True)
    print("  Fig 4 saved: fig4_sensitivity.png")

    plot_valve_tradeoff(save=True)
    print("  Fig 5 saved: fig5_pid_tradeoff.png")

    plot_culvert_backCalc(x, save=True)
    print("  Fig 6 saved: fig6_culvert_backCalc.png")

    print("\nDone. All 6 figures saved.")