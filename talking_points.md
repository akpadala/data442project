# DATA 442 Final Project — Individual Presentation Talking Points
## Meghna — Panama Canal Lock Cycle Simulation

---

## What I personally implemented

Each item below is something you can point to directly if asked "what did you build?"

### `parameters.py`
All physical constants for the simulation, centralized in one file so that calibration sweeps and sensitivity analyses can be run by changing a single value rather than hunting through simulation logic. Key design decision: derived quantities (INERTANCE, RESISTANCE) are computed from primitive inputs (diameter, length, density) so the file is self-documenting and physically traceable.

### `dynamics.py` (original + both bug fixes)
The physics engine. Defines the ODE right-hand sides for all four FSM phases using `PHASE_CONFIG` — a dispatch table mapping phase index to which state vector elements are target and fill. Also defines `H_sea(t)` (tidal forcing), `valve_area(t, t_entry)` (the ramp function), and the unified `system_dynamics` function.

Two bugs were introduced in the partner's version and corrected:
- **Bug 1 — missing `RHO*G` scaling:** The water hammer driving term was `delta_h` instead of `RHO*G*delta_h`. This is a dimensional inconsistency (meters vs pressure) that made `dQ/dt` ~9810× too small, causing the simulation to produce near-zero flow. Fix: restore `RHO * G * delta_h` in the numerator.
- **Bug 2 — inverted basin index mapping:** `BASIN_INDICES = [3,2,1,0]` caused WSB_DRAIN_1 to fill basin 1 (index 3 in the state vector) instead of basin 3 (index 1). The state vector is `[H_L, H_B3, H_B2, H_B1, Q]`, so filling basin 3 requires updating `y[1]`, not `y[3]`. Fix: replace with `PHASE_CONFIG` which maps phase index to state vector index explicitly.

### `simulation.py` (original + basin index fix)
The FSM integration loop. Calls `solve_ivp` once per phase with event detection to stop at `|ΔH| < ε`. Key design decision: rather than solving the full simulation in one call, each FSM phase is a separate `solve_ivp` call with the final state as the initial condition for the next phase, and `Q` reset to zero between phases to model valve closing. This is what makes the FSM switching numerically clean.

### `analysis.py` — Figures 1–6
All six original analysis figures:
- **Fig 1** — Water levels over time (primary simulation result)
- **Fig 2** — Phase portrait ΔH vs Q (state-space trajectory, analogous to displacement vs velocity)
- **Fig 3** — Water recovery bar chart (validates against ACP 60% spec)
- **Fig 4** — Sensitivity analysis on Cd (parameter uncertainty)
- **Fig 5** — Valve timing tradeoff (throughput vs peak flow)
- **Fig 6** — Culvert back-calculation from published velocity spec

### `transfer_function.py` — Figures 7–9 (Task 1)
Formal Laplace analysis of the FINAL_DRAIN phase. Derives G1(s) = Q(s)/ΔH(s) and G2(s) = H_L(s)/H_sea(s). Generates Bode plot, step response, and analytical overlay on simulation. Key result: tidal frequency is 40× below natural frequency, placing the system in the low-frequency regime where gain ≈ 1 — the lock tracks the tide with near-unity amplitude.

### `step_response_analysis.py` — Figure 10 (Task 2)
Computes the damping ratio analytically (ζ = 0.0019, highly underdamped) and verifies the second-order character of Q. Key result: natural period is 10.7 min but drain completes in 5.1 min — less than half a period — so water hammer oscillations cannot develop. The canal geometry itself provides inherent protection; the valve ramp is an additional safety margin.

### `pid_controller.py` — Figure 11 (Task 3)
Closed-loop PID valve controller regulating the valve fraction u(t) ∈ [0,1] to track a linear ΔH setpoint while satisfying the 8 m/s velocity constraint. Tuned via Kp sweep. Key result: PID satisfies velocity spec (7.2 m/s vs 19.3 m/s uncontrolled) at the cost of 183 extra seconds per drain phase — quantifiable throughput cost of the safety constraint.

### `fig_fsm_diagram.py` — FSM diagram (Task 4)
Presentation-quality Stateflow-style FSM diagram showing all 8 states, mathematical transition conditions, and active ODEs in each hydraulic state.

### `fig_validation.py` — Validation figure (Task 5)
Honest validation against published paper figures. No-basin timing matches to 3.1%. With-basin discrepancy (46%) is explained and attributed to simplified rectangular basin geometry without depth limits.

### `visualize.py`
Real-time animated simulation using matplotlib FuncAnimation. Artist-update pattern (no `ax.cla()` per frame) for smooth rendering. Implements all five tasks: SimulationState class, cross-section animation, live time series, FSM state indicator with flash on transition, and interactive sliders for Cd, valve time, and sim speed.

---

## One-sentence technical justification per file

| File | Design decision |
|---|---|
| `parameters.py` | Centralized constants enable parameter sweeps without touching simulation logic |
| `dynamics.py` | Single `system_dynamics` function with `PHASE_CONFIG` dispatch avoids code duplication across four phases |
| `simulation.py` | Per-phase `solve_ivp` with event detection gives numerically clean FSM switching |
| `analysis.py` | Six independent figures answer six independent questions about the system |
| `transfer_function.py` | `scipy.signal` provides transfer function, Bode, and step response with no manual Laplace algebra in code |
| `step_response_analysis.py` | Analytical ζ computation verified against simulation — two independent methods, same result |
| `pid_controller.py` | Hard velocity constraint enforced inside PID output rather than as a soft penalty — guarantees spec satisfaction |
| `fig_fsm_diagram.py` | Pure matplotlib (no external tools) means the diagram is reproducible from source |
| `fig_validation.py` | Gate overhead shown as an explicit separate bar — not folded into hydraulic simulation |
| `visualize.py` | Artist-update pattern instead of `cla()` reduces per-frame cost by ~10–20× |

---

## The two bug fixes — how to frame them

If asked about collaboration or the bugs, use this framing — accurate and constructive:

> "During integration testing I identified two issues in the shared dynamics file. The first was a dimensional inconsistency in the water hammer driving term — the head difference in meters needed to be scaled by ρg to convert to pressure before dividing by inertance. The second was an index mismatch between the phase labels (basin 3 first) and the state vector layout (H_B3 is at index 1, not 3). Both were straightforward to identify by checking units and running the simulation to verify that water levels changed as expected."

Do not say "my partner's bugs." Say "issues I found during integration."

---

## Answers to three likely professor questions

### Q1: "Why Euler integration in the visualizer instead of RK45?"

> "The visualizer uses forward Euler with dt=1s for real-time performance. Euler is O(dt) accurate but fast enough per step to maintain 25fps. The simulation figures use `solve_ivp` with RK45 (same as MATLAB's `ode45`) which is O(dt⁵) accurate with adaptive step size. We verified that both produce consistent trajectories over the 21-minute cycle — the Euler visualizer is slightly less accurate but sufficient for the visual demonstration. Using RK45 in the visualizer would require running the full simulation ahead of time and replaying it, which would lose the interactive parameter adjustment."

### Q2: "What does the pole of your transfer function tell you about the system?"

> "G1(s) = Q(s)/ΔH(s) has a single pole at s = −R/L = −0.000037 rad/s, giving a time constant τ = L/R ≈ 449 minutes. This means flow takes over 7 hours to fully decay — much longer than a single lock cycle. Physically this makes sense: the culverts have very low hydraulic resistance relative to their inertance, so once flow is established it persists. G2(s) = H_L(s)/H_sea(s) has two complex conjugate poles with natural frequency ωn = 0.0073 rad/s and damping ratio ζ = 0.0025. The poles are in the left half-plane confirming stability. The tidal forcing frequency is 40× lower than ωn, placing the system in the low-frequency regime where the lock level tracks the tide with near-unity gain — which is why tidal phase affects drain time."

### Q3: "How did you validate your model?"

> "We validated against two published figures from a Panama Canal engineering paper. The no-basin drain time — which tests the core hydraulic model without basin geometry — matches to within 3.1% once gate mechanical overhead is accounted for. This validates the Torricelli orifice flow combined with the water hammer inertance ODE. The with-basin cycle time is slower in our model by about 46%, which we attribute to simplified rectangular basin geometry — our basins have no depth limit, so they over-fill. Both discrepancies are consistent: the same geometric simplification explains both the timing error and the 68.7% vs 60% over-recovery. The core hydraulics are validated; the basin geometry is a known, explained limitation."

---

## Your 2-minute individual contribution speech

Practice saying this out loud. Aim for 90–120 seconds:

> "My contributions covered the full modeling pipeline. I wrote `parameters.py` to centralize all physical constants, `dynamics.py` with the water hammer ODEs and valve ramp, and `simulation.py` with the FSM integration loop using event-driven phase switching. During integration I identified and fixed two bugs: a dimensional inconsistency in the driving term — missing a ρg factor — and an inverted index mapping that was filling the wrong basin in each phase. Both were caught by checking units and verifying simulation outputs.
>
> For analysis I generated six figures including a phase portrait, sensitivity analysis, and culvert back-calculation from a published velocity spec. For the new work this week: I derived the transfer function for the FINAL_DRAIN phase, showed that the system is underdamped but completes its drain before one natural period — so water hammer can't develop. I implemented a closed-loop PID controller that satisfies the 8 m/s velocity spec at the cost of 183 extra seconds per phase. And I built the real-time animated visualizer with interactive parameter sliders.
>
> The part I'm most proud of is the validation figure — it's honest about where the model agrees and where it doesn't, and explains why. The no-basin timing matches to 3.1%. The with-basin discrepancy is real and attributed to a specific, correctable limitation in the basin geometry."
