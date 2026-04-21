"""
parameters.py
All physical constants for the Panama Canal lock simulation.
Real-world values sourced from Panama Canal Authority specifications.
"""

import numpy as np

# ── Gravitational constant ────────────────────────────────────────────────────
G = 9.81  # m/s²

# ── Lock chamber geometry (Neopanamax) ────────────────────────────────────────
LOCK_LENGTH    = 427.0   # m
LOCK_WIDTH     = 55.0    # m
LOCK_AREA      = LOCK_LENGTH * LOCK_WIDTH  # m²  (~23,485 m²)

LOCK_H_UPPER   = 26.0    # m  — full water height when lock is at lake level
LOCK_H_LOWER   = 0.0     # m  — empty (sea level reference)

# ── Water saving basin geometry ───────────────────────────────────────────────
# Three basins per chamber, each ~70m wide × 5.5m deep
# Basin surface areas approximate — basins are shallower and wider than the lock
BASIN_LENGTH   = 427.0   # m  (same length as lock chamber)
BASIN_WIDTH    = 70.0    # m
BASIN_AREA     = BASIN_LENGTH * BASIN_WIDTH  # m²  (~29,890 m²)

# Initial water heights for each basin (at start of simulation)
# Basin 3 starts at 2/3 lock height, basin 2 at 1/3, basin 1 near zero
BASIN_3_H_INIT = 18.0    # m  (highest basin)
BASIN_2_H_INIT = 10.0    # m  (mid basin)
BASIN_1_H_INIT = 4.0     # m  (lowest basin)

# ── Culvert / valve parameters ────────────────────────────────────────────────
CULVERT_DIAMETER = 7.0          # m  (approximate — Neopanamax culverts are massive)
CULVERT_AREA     = np.pi * (CULVERT_DIAMETER / 2) ** 2  # m²
CULVERT_LENGTH   = 300.0        # m  (approximate path through lock walls)

CD               = 0.7          # discharge coefficient (dimensionless)
VALVE_OPEN_TIME  = 240.0        # s  (valve fully open after ~4 minutes)
A_VALVE_MAX      = CULVERT_AREA # m²  (fully open = full culvert area)

# ── Inertance and resistance (water hammer model) ─────────────────────────────
RHO              = 1000.0       # kg/m³ (freshwater density)
DARCY_FRICTION   = 0.02         # f  (Darcy-Weisbach friction factor, smooth concrete)

# Inertance L = ρ * l / A_c   (kg/m⁴)
INERTANCE = RHO * CULVERT_LENGTH / CULVERT_AREA

# Hydraulic resistance R = ρ * f * l / (2 * D * A_c²)   (kg/m⁷)
RESISTANCE = (RHO * DARCY_FRICTION * CULVERT_LENGTH) / (2 * CULVERT_DIAMETER * CULVERT_AREA ** 2)

# ── Tidal forcing (Pacific side) ──────────────────────────────────────────────
TIDE_AMPLITUDE   = 3.0          # m  (Pacific tidal range up to 6m, using half-amplitude)
TIDE_PERIOD      = 6 * 3600.0   # s  (~6 hour semidiurnal tidal period)
TIDE_OMEGA       = 2 * np.pi / TIDE_PERIOD  # rad/s
TIDE_PHASE       = 0.0          # rad (can adjust to start at high/low tide)
TIDE_MEAN        = 0.0          # m  (mean sea level reference)

# ── FSM transition threshold ──────────────────────────────────────────────────
EPSILON          = 0.05         # m  (equalization tolerance — switch state when |ΔH| < ε)

# ── Simulation time ───────────────────────────────────────────────────────────
T_MAX            = 6 * 3600.0   # s  (simulate 6 hours — one full tidal cycle)
DT_MAX           = 1.0          # s  (max solver step size)
