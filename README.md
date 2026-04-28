# Panama Canal Lock Cycle Simulation
**DATA 442 — System Design and Engineering**
**University of North Carolina at Chapel Hill, Spring 2026**

**Authors:** Mihika Tyagi, Akshita Padala
**Reference:** Calvo Gobbetti, L. E. (2013). Design of the filling and emptying system of the new Panama Canal locks. *Journal of Applied Water Engineering and Research*, *1*(1), 28–38. https://doi.org/10.1080/23249676.2013.827899

---

## Project Overview

This project models the hydraulic lock cycle of the Neopanamax Panama Canal locks (Cocoli/Agua Clara) as a **switched dynamical system** — a hybrid system where discrete FSM logic governs which coupled ODEs are active at each phase.

A ship lowering from Gatun Lake level (26m) to sea level (0m) passes through four hydraulic phases:
1. **WSB Drain 1** — lock drains into Water Saving Basin 3 (highest)
2. **WSB Drain 2** — lock drains into Water Saving Basin 2 (mid)
3. **WSB Drain 3** — lock drains into Water Saving Basin 1 (lowest)
4. **Final Drain** — lock drains to tidal sea (water lost, sinusoidal forcing)

The FSM transitions fire when the active water pair equalizes: |ΔH| < ε = 0.05m.

---

## System Model

### Core ODEs (active in each hydraulic phase)

**Water hammer ODE:**
```
L · dQ/dt + R·Q = ρg · ΔH · Cd · (Av/Amax)
```

**Lock level:**
```
dH_L/dt = -Q / A_L
```

**Active basin (only one per phase):**
```
dH_Bi/dt = +Q / A_B
```

**Valve ramp (non-linear input):**
```
Av(t) = Amax · min(t / T_open, 1)
```

**Tidal forcing (FINAL_DRAIN only):**
```
H_sea(t) = A · sin(ω_tide · t)
```

### State vector
```
x = [H_L, H_B3, H_B2, H_B1, Q]
```
The state vector is continuous across FSM transitions. Only Q resets to zero (valve closes between phases).

### Transfer function (derived analytically)
```
G2(s) = H_L(s)/H_sea(s) = ωn² / (s² + 2ζωn·s + ωn²)
```
- ωn = 7.32 × 10⁻³ rad/s (natural period = 14.3 min)
- ζ = 0.0025 (underdamped)
- ω_tide/ωn = 0.040 → lock tracks tide with near-unity gain

---

## Code Files

### `parameters.py`
All physical constants for the simulation. Every other file imports from here. Changing a value here propagates through the entire simulation automatically.

Key parameters:
- Lock geometry: 427m × 55m, H_upper = 26m
- Basin geometry: 427m × 70m
- Culvert: D = 7m, L = 300m, Cd = 0.7
- Inertance: L = ρl/A_c (hydraulic analog of mass)
- Resistance: R = ρfl/(2DA_c²) (hydraulic analog of damping)
- Tidal forcing: A = 3m, T = 6hr (Pacific side)

*Author: Mihika Tyagi*

---

### `dynamics.py`
The physics engine. Defines the ODE right-hand sides for all four FSM phases via a single `system_dynamics(t, y, phase_idx, t_entry)` function using a `PHASE_CONFIG` dispatch table.

Also defines:
- `H_sea(t)` — tidal forcing function
- `valve_area(t, t_entry)` — linear valve ramp

*Authors: Mihika Tyagi (original + final corrected version), Akshita [Last Name] (intermediate rewrite)*

---

### `simulation.py`
FSM integration loop. Calls `scipy.solve_ivp` once per phase with event detection to stop when |ΔH| < ε. The final state of each phase is the initial condition for the next. Q resets to zero between phases.

Key function: `run_lock_cycle()` returns `(t, x, log)` where log records phase entry/exit times.

*Authors: Mihika Tyagi (original + final corrected version), Akshita [Last Name] (intermediate rewrite)*

---

### `analysis.py`
Generates Figures 1–6. Run with `python analysis.py`.

*Author: Mihika Tyagi*

---

### `transfer_function.py`
Derives G1(s) and G2(s) analytically from the FINAL_DRAIN ODEs using Laplace transforms. Generates Bode plot, step response, and analytical validation overlay on simulation output. Prints full derivation to terminal.

Run with: `python transfer_function.py`

*Author: Mihika Tyagi*

---

### `step_response_analysis.py`
Computes damping ratio ζ analytically for WSB and FINAL_DRAIN phases. Verifies second-order character by running step vs ramp valve simulations. Shows that drain completes in less than one natural period — water hammer cannot develop.

Run with: `python step_response_analysis.py`

*Author: Mihika Tyagi*

---

### `pid_controller.py`
Implements a closed-loop PID controller regulating valve fraction u(t) ∈ [0,1] to track a linear ΔH setpoint while enforcing the 8 m/s velocity constraint from Calvo Gobbetti (2013). Includes tuning sweep and before/after comparison.

Run with: `python pid_controller.py`

*Author: Mihika Tyagi*

---

### `fig_fsm_diagram.py`
Generates a presentation-quality Stateflow-style FSM diagram showing all 8 states, mathematical transition conditions, and active ODEs in each hydraulic state box.

Run with: `python fig_fsm_diagram.py`

*Author: Mihika Tyagi*

---

### `fig_validation.py`
Validates simulation results against Calvo Gobbetti (2013). Shows timing comparison (hydraulic only vs hydraulic + gate overhead vs paper figures) and water recovery comparison vs ACP 60% spec. Honestly accounts for known modeling limitations.

Run with: `python fig_validation.py`

*Author: Mihika Tyagi*

---

### `visualize.py`
Real-time animated simulation using matplotlib FuncAnimation. Uses artist-update pattern (no ax.cla() per frame) for smooth rendering at 25fps.

Controls:
- `SPACE` — pause / resume
- `R` — reset to initial conditions
- `↑ / ↓` — increase / decrease sim speed
- `Q` / `Escape` — quit

Sliders: Cd (discharge coefficient), valve open time, sim speed.

Run with: `python visualize.py`

*Author: Mihika Tyagi*

---

## Figures

### Presentation figures

| Figure | File | Description |
|---|---|---|
| FSM Diagram | `fig_fsm_diagram.png` | Stateflow-style diagram — 8 states, transition conditions, active ODEs |
| Bode Plot | `fig7_bode.png` | G2(s) frequency response — shows tidal frequency in low-gain regime |
| Step Response | `fig8_step_response.png` | G2(s) step response — underdamped, natural period 14.3 min |
| Q Transient | `fig10_q_transient.png` | Step vs ramp valve, damping ratio, water hammer analysis |
| PID Comparison | `fig11_pid_comparison.png` | Controlled vs uncontrolled valve — velocity constraint enforcement |
| Water Levels | `fig1_water_levels.png` | Full 21-min cycle — lock level, basin levels, flow rate, tidal input |
| Validation | `fig_validation.png` | Timing and recovery vs Calvo Gobbetti (2013) |

### Backup / supplementary figures

| Figure | File | Description | When to use |
|---|---|---|---|
| Phase Portrait | `fig2_phase_portrait.png` | ΔH vs Q trajectory — state space analog of displacement vs velocity | If asked about state space behavior |
| TF Validation | `fig9_tf_validation.png` | Analytical G2(s) overlaid on ODE simulation | If asked how transfer function was verified |
| Sensitivity | `fig4_sensitivity.png` | Cd vs cycle time — parameter uncertainty analysis | If asked about parameter assumptions |
| Water Recovery | `fig3_water_recovery.png` | Basin volume breakdown — 68.7% recovery | Redundant with validation figure |
| Valve Tradeoff | `fig5_pid_tradeoff.png` | Valve timing vs cycle time and peak Q | Superseded by fig11 |
| Culvert Calc | `fig6_culvert_backCalc.png` | Culvert diameter back-calculated from velocity spec | If asked about culvert parameters |

---

## How to run

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

Run full simulation and generate all analysis figures:
```bash
python analysis.py
python transfer_function.py
python step_response_analysis.py
python pid_controller.py
python fig_fsm_diagram.py
python fig_validation.py
```

Run real-time visualizer:
```bash
python visualize.py
```

---

## Key results

| Metric | Value | Source |
|---|---|---|
| Total cycle time | 21.3 min (hydraulic) | Simulation |
| No-basin drain | 6.2 min hydraulic / 9.7 min + gates | Simulation |
| Paper no-basin | 10 min | Calvo Gobbetti (2013) |
| Agreement | 3.1% error | — |
| Water recovery | 68.7% | Simulation |
| ACP spec | 60% | Calvo Gobbetti (2013) |
| Natural frequency | ωn = 7.32 × 10⁻³ rad/s | Transfer function |
| Damping ratio | ζ = 0.0025 (underdamped) | Analytical |
| PID peak velocity | 7.2 m/s (satisfies 8 m/s spec) | PID simulation |

---

## Known limitations

1. **Basin geometry simplified** — rectangular basins with no depth limit. Real basins stop filling at a bounded depth. This causes over-recovery (68.7% vs 60%) and slower with-basin cycle time.
2. **Single equivalent culvert** — model uses one culvert of equivalent area. Real locks use multiple parallel culverts. This affects velocity calculations.
3. **Gate mechanics not modeled** — gate closing/opening (~3.5 min) is outside model scope. Adding gate overhead brings no-basin timing to within 3.1% of paper.
4. **Euler integration in visualizer** — real-time visualizer uses forward Euler (dt=1s) for performance. Analysis figures use RK45 via scipy.solve_ivp.