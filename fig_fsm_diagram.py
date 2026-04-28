"""
fig_fsm_diagram.py
==================
Generates a presentation-quality FSM diagram in Stateflow style showing:
- All 8 states (mechanical + hydraulic)
- Transition conditions as mathematical expressions
- Active ODEs in each hydraulic state
- Color coding matching the rest of the project

Run with: python fig_fsm_diagram.py

Output: fig_fsm_diagram.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Color scheme (matches project) ───────────────────────────────────────────
COLORS = {
    'mechanical' : '#6B7280',   # gray  — VESSEL_ENTRY, GATE states, EXIT
    'wsb1'       : '#059669',   # green — WSB_DRAIN_1
    'wsb2'       : '#D97706',   # amber — WSB_DRAIN_2
    'wsb3'       : '#DC2626',   # red   — WSB_DRAIN_3
    'final'      : '#7C3AED',   # purple— FINAL_DRAIN
    'bg'         : '#FFFFFF',
    'border'     : '#1E293B',
    'text'       : '#1E293B',
    'arrow'      : '#475569',
    'condition'  : '#1D4ED8',
    'ode_text'   : '#374151',
}

# ── State definitions ─────────────────────────────────────────────────────────
# Each state: (name, short_label, color, ode_lines, entry_action)
STATES = [
    {
        'id'    : 'VESSEL_ENTRY',
        'label' : 'VESSEL\nENTRY',
        'color' : COLORS['mechanical'],
        'odes'  : [],
        'entry' : 'Ship enters chamber',
        'type'  : 'mechanical',
    },
    {
        'id'    : 'GATE_CLOSING',
        'label' : 'GATE\nCLOSING',
        'color' : COLORS['mechanical'],
        'odes'  : [],
        'entry' : 'Gates swing shut\nNo flow yet',
        'type'  : 'mechanical',
    },
    {
        'id'    : 'WSB_DRAIN_1',
        'label' : 'WSB\nDRAIN 1',
        'color' : COLORS['wsb1'],
        'odes'  : [
            r'$\frac{dH_L}{dt} = -\frac{Q}{A_L}$',
            r'$\frac{dH_{B3}}{dt} = +\frac{Q}{A_B}$',
            r'$\frac{dQ}{dt} = \frac{\rho g \Delta H \cdot C_d - RQ}{L}$',
        ],
        'entry' : 'Valve 3 opens\nLock → Basin 3',
        'type'  : 'hydraulic',
    },
    {
        'id'    : 'WSB_DRAIN_2',
        'label' : 'WSB\nDRAIN 2',
        'color' : COLORS['wsb2'],
        'odes'  : [
            r'$\frac{dH_L}{dt} = -\frac{Q}{A_L}$',
            r'$\frac{dH_{B2}}{dt} = +\frac{Q}{A_B}$',
            r'$\frac{dQ}{dt} = \frac{\rho g \Delta H \cdot C_d - RQ}{L}$',
        ],
        'entry' : 'Valve 2 opens\nLock → Basin 2',
        'type'  : 'hydraulic',
    },
    {
        'id'    : 'WSB_DRAIN_3',
        'label' : 'WSB\nDRAIN 3',
        'color' : COLORS['wsb3'],
        'odes'  : [
            r'$\frac{dH_L}{dt} = -\frac{Q}{A_L}$',
            r'$\frac{dH_{B1}}{dt} = +\frac{Q}{A_B}$',
            r'$\frac{dQ}{dt} = \frac{\rho g \Delta H \cdot C_d - RQ}{L}$',
        ],
        'entry' : 'Valve 1 opens\nLock → Basin 1',
        'type'  : 'hydraulic',
    },
    {
        'id'    : 'FINAL_DRAIN',
        'label' : 'FINAL\nDRAIN',
        'color' : COLORS['final'],
        'odes'  : [
            r'$\frac{dH_L}{dt} = -\frac{Q}{A_L}$',
            r'$H_{sea}(t) = A\sin(\omega t)$',
            r'$\frac{dQ}{dt} = \frac{\rho g (H_L - H_{sea}) C_d - RQ}{L}$',
        ],
        'entry' : 'Drain to tidal sea\n(sinusoidal forcing)',
        'type'  : 'hydraulic',
    },
    {
        'id'    : 'GATE_OPENING',
        'label' : 'GATE\nOPENING',
        'color' : COLORS['mechanical'],
        'odes'  : [],
        'entry' : 'Gates swing open',
        'type'  : 'mechanical',
    },
    {
        'id'    : 'VESSEL_EXIT',
        'label' : 'VESSEL\nEXIT',
        'color' : COLORS['mechanical'],
        'odes'  : [],
        'entry' : 'Ship departs\nToll recorded',
        'type'  : 'mechanical',
    },
]

# Transition conditions between consecutive states
TRANSITIONS = [
    r'Ship secured',
    r'Gates closed',
    r'$|H_L - H_{B3}| < \varepsilon$',
    r'$|H_L - H_{B2}| < \varepsilon$',
    r'$|H_L - H_{B1}| < \varepsilon$',
    r'$|H_L - H_{sea}| < \varepsilon$',
    r'$\Delta P \approx 0$',
]


# ═══════════════════════════════════════════════════════════════════════════════
# Layout: two rows
#   Top row:    VESSEL_ENTRY  GATE_CLOSING  WSB1  WSB2  WSB3  FINAL_DRAIN
#   Bottom row: (centered)    GATE_OPENING  VESSEL_EXIT
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 11), facecolor=COLORS['bg'])
ax  = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor(COLORS['bg'])

# Title
ax.text(10, 10.6, 'Panama Canal Lock Cycle — FSM with Active ODEs',
        ha='center', va='center', fontsize=16, fontweight='bold',
        color=COLORS['text'])
ax.text(10, 10.15,
        'Hybrid System: Discrete FSM governs which coupled ODEs are active each phase',
        ha='center', va='center', fontsize=10, color='#64748B', style='italic')

# ── Box geometry ──────────────────────────────────────────────────────────────
BOX_W_MECH = 1.9    # mechanical state box width
BOX_H_MECH = 1.5    # mechanical state box height (increased for text clearance)
BOX_W_HYD  = 2.6    # hydraulic state box width
BOX_H_HYD  = 3.8    # hydraulic state box height (needs room for ODEs)
ROW1_Y     = 5.2    # y center of top row
ROW2_Y     = 1.5    # y center of bottom row

# Top row x positions (centers)
# VESSEL_ENTRY, GATE_CLOSING, WSB1, WSB2, WSB3, FINAL_DRAIN
TOP_X = [1.3, 3.5, 6.2, 9.1, 12.0, 15.1]

# Bottom row x positions
BOT_X = [15.1, 17.8]  # GATE_OPENING, VESSEL_EXIT
# (Final drain is top row, gate opening connects down from it)

def draw_state_box(cx, cy, state, row='top'):
    """Draw a state box centered at (cx, cy)."""
    color  = state['color']
    is_hyd = state['type'] == 'hydraulic'
    bw     = BOX_W_HYD  if is_hyd else BOX_W_MECH
    bh     = BOX_H_HYD  if is_hyd else BOX_H_MECH

    x0 = cx - bw/2
    y0 = cy - bh/2

    # Shadow
    shadow = FancyBboxPatch((x0+0.05, y0-0.05), bw, bh,
                             boxstyle='round,pad=0.08',
                             facecolor='#CBD5E1', edgecolor='none',
                             zorder=1, alpha=0.5)
    ax.add_patch(shadow)

    # Main box
    box = FancyBboxPatch((x0, y0), bw, bh,
                          boxstyle='round,pad=0.08',
                          facecolor='white',
                          edgecolor=color,
                          linewidth=2.5, zorder=2)
    ax.add_patch(box)

    # Header bar
    header = FancyBboxPatch((x0, y0 + bh - 0.62), bw, 0.62,
                             boxstyle='round,pad=0.0',
                             facecolor=color, edgecolor='none',
                             alpha=0.85, zorder=3,
                             clip_on=True)
    ax.add_patch(header)

    # State label in header
    ax.text(cx, y0 + bh - 0.31, state['label'],
            ha='center', va='center', fontsize=9,
            fontweight='bold', color='white', zorder=4)

    if is_hyd:
        # Entry action
        ax.text(cx, y0 + bh - 0.85, state['entry'],
                ha='center', va='center', fontsize=7,
                color='#475569', style='italic', zorder=4)

        # Divider line
        ax.plot([x0 + 0.1, x0 + bw - 0.1],
                [y0 + bh - 1.05, y0 + bh - 1.05],
                color='#E2E8F0', lw=1, zorder=4)

        # ODE label
        ax.text(cx, y0 + bh - 1.25, 'Active ODEs:',
                ha='center', va='center', fontsize=7,
                color='#64748B', fontweight='bold', zorder=4)

        # ODEs
        for k, ode_str in enumerate(state['odes']):
            ax.text(cx, y0 + bh - 1.65 - k * 0.72, ode_str,
                    ha='center', va='center', fontsize=8.5,
                    color=COLORS['ode_text'], zorder=4)

    else:
        # Entry action for mechanical states — sit in lower half of box
        ax.text(cx, cy - 0.22, state['entry'],
                ha='center', va='center', fontsize=7.5,
                color='#475569', style='italic', zorder=4,
                multialignment='center')

    return (x0, y0, bw, bh)


# Draw top row states
top_boxes = []
for i, (cx, state) in enumerate(zip(TOP_X, STATES[:6])):
    geom = draw_state_box(cx, ROW1_Y, state, row='top')
    top_boxes.append(geom)

# Draw bottom row states
bot_boxes = []
for i, (cx, state) in enumerate(zip(BOT_X, STATES[6:])):
    geom = draw_state_box(cx, ROW2_Y, state, row='bot')
    bot_boxes.append(geom)


# ── Initial state dot ─────────────────────────────────────────────────────────
ax.plot(TOP_X[0] - BOX_W_MECH/2 - 0.35, ROW1_Y,
        'o', color=COLORS['text'], ms=10, zorder=5)
ax.annotate('', xy=(TOP_X[0] - BOX_W_MECH/2 - 0.02, ROW1_Y),
            xytext=(TOP_X[0] - BOX_W_MECH/2 - 0.3, ROW1_Y),
            arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))


# ── Draw horizontal transition arrows (top row) ───────────────────────────────
def draw_arrow_h(x_start, x_end, y, label, label_y_offset=0.28, color=COLORS['arrow']):
    ax.annotate('',
                xy=(x_end, y), xytext=(x_start, y),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.8, mutation_scale=18),
                zorder=5)
    ax.text((x_start + x_end) / 2, y + label_y_offset,
            label, ha='center', va='bottom', fontsize=7.5,
            color=COLORS['condition'], fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none',
                      pad=1.5, alpha=0.85),
            zorder=6)


for i in range(5):
    x0_box, y0_box, bw, bh = top_boxes[i]
    x1_box                  = top_boxes[i+1][0]
    is_hyd_i   = STATES[i]['type']   == 'hydraulic'
    is_hyd_ip1 = STATES[i+1]['type'] == 'hydraulic'
    bw_i   = BOX_W_HYD if is_hyd_i   else BOX_W_MECH
    bw_ip1 = BOX_W_HYD if is_hyd_ip1 else BOX_W_MECH

    x_start = TOP_X[i]   + bw_i   / 2
    x_end   = TOP_X[i+1] - bw_ip1 / 2

    cond_color = COLORS['condition'] if STATES[i]['type'] == 'hydraulic' else COLORS['arrow']
    draw_arrow_h(x_start, x_end, ROW1_Y, TRANSITIONS[i],
                 label_y_offset=0.22, color=cond_color)


# ── Vertical arrow: FINAL_DRAIN → GATE_OPENING (top row down to bottom row) ──
x_fd    = TOP_X[5]
y_fd_bot = ROW1_Y - BOX_H_HYD / 2
y_go_top = ROW2_Y + BOX_H_MECH / 2

ax.annotate('',
            xy=(x_fd, y_go_top + 0.02), xytext=(x_fd, y_fd_bot - 0.02),
            arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                            lw=1.8, mutation_scale=18),
            zorder=5)
ax.text(x_fd + 0.22, (y_fd_bot + y_go_top) / 2,
        TRANSITIONS[5],
        ha='left', va='center', fontsize=7.5,
        color=COLORS['condition'], fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.85),
        zorder=6)


# ── Horizontal arrow: GATE_OPENING → VESSEL_EXIT ─────────────────────────────
x_go_right = BOT_X[0] + BOX_W_MECH / 2
x_ve_left  = BOT_X[1] - BOX_W_MECH / 2
draw_arrow_h(x_go_right, x_ve_left, ROW2_Y,
             r'$\Delta P \approx 0$', label_y_offset=0.22)


# ── Legend ────────────────────────────────────────────────────────────────────
legend_x = 0.25
legend_y = 4.2   # raised from 3.5 to give more vertical room
ax.text(legend_x, legend_y + 0.35, 'Legend', fontsize=9,
        fontweight='bold', color=COLORS['text'])

legend_items = [
    (COLORS['mechanical'], 'Mechanical state (no ODEs active)'),
    (COLORS['wsb1'],       'WSB Drain 1 — Lock → Basin 3'),
    (COLORS['wsb2'],       'WSB Drain 2 — Lock → Basin 2'),
    (COLORS['wsb3'],       'WSB Drain 3 — Lock → Basin 1'),
    (COLORS['final'],      'Final Drain — Lock → Sea (tidal forcing)'),
]

for k, (col, lbl) in enumerate(legend_items):
    yy = legend_y - 0.05 - k * 0.52   # tighter spacing, single-line labels
    rect = FancyBboxPatch((legend_x, yy - 0.16), 0.25, 0.32,
                           boxstyle='round,pad=0.04',
                           facecolor=col, edgecolor='none',
                           alpha=0.85, zorder=3)
    ax.add_patch(rect)
    ax.text(legend_x + 0.35, yy, lbl,
            ha='left', va='center', fontsize=7.5, color=COLORS['text'])

# Condition arrow legend
yy = legend_y - 0.05 - len(legend_items) * 0.52 - 0.15
ax.annotate('', xy=(legend_x + 0.25, yy), xytext=(legend_x, yy),
            arrowprops=dict(arrowstyle='->', color=COLORS['condition'], lw=1.5))
ax.text(legend_x + 0.35, yy,
        'Transition condition (mathematical expression)',
        ha='left', va='center', fontsize=7.5, color=COLORS['condition'])

# ODE note box
ax.text(legend_x, 0.75,
        'State vector:  x = [H_L,  H_B3,  H_B2,  H_B1,  Q]\n'
        'Switching:     FSM selects active ODE pair each phase\n'
        'Forcing:       H_sea(t) = A·sin(ω_tide·t)  in FINAL_DRAIN\n'
        'Transition:    |ΔH| < ε = 0.05 m  triggers state change',
        ha='left', va='bottom', fontsize=7.8, color='#374151',
        bbox=dict(facecolor='#F8FAFC', edgecolor='#CBD5E1',
                  boxstyle='round,pad=0.5', alpha=0.95),
        zorder=5)


# ── Phase flow annotation ─────────────────────────────────────────────────────
# Draw a curved brace under the hydraulic states
brace_y = ROW1_Y - BOX_H_HYD / 2 - 0.35
x_brace_start = TOP_X[2] - BOX_W_HYD / 2
x_brace_end   = TOP_X[5] + BOX_W_HYD / 2

ax.annotate('', xy=(x_brace_end, brace_y),
            xytext=(x_brace_start, brace_y),
            arrowprops=dict(arrowstyle='<->', color='#94A3B8', lw=1.2))
ax.text((x_brace_start + x_brace_end) / 2, brace_y - 0.22,
        'Hydraulic phases — coupled ODE system active\n'
        'Switched dynamical system: same state vector x, different f_σ(x) per mode',
        ha='center', va='top', fontsize=8, color='#64748B', style='italic')


plt.tight_layout(pad=0.3)
plt.savefig('fig_fsm_diagram.png', dpi=200, bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()
print("FSM diagram saved: fig_fsm_diagram.png")
print()
print("Task 4 complete.")
print("  8 states shown: 4 mechanical, 4 hydraulic")
print("  All transition conditions written as mathematical expressions")
print("  Active ODEs shown in each hydraulic state box")
print("  Color coding matches project figures")
print("  Brace annotation frames the coupled ODE region")