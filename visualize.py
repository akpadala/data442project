"""
visualize.py — Panama Canal real-time visual simulation
Run with: python visualize.py

Controls:
  SPACE       pause / resume
  r           reset to initial conditions
  ↑ / ↓       increase / decrease sim speed
  q / Escape  quit
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon
import matplotlib.patheffects as pe

import parameters as P
from dynamics import (
    system_dynamics, H_sea, valve_area,
    STATE_NAMES, PHASE_CONFIG,
    WSB_DRAIN_1, WSB_DRAIN_2, WSB_DRAIN_3, FINAL_DRAIN
)
from parameters import (
    LOCK_H_UPPER, BASIN_3_H_INIT, BASIN_2_H_INIT, BASIN_1_H_INIT
)

# ── Color scheme ──────────────────────────────────────────────────────────────
C = {
    'lock'    : '#2563EB',
    'basin3'  : '#059669',
    'basin2'  : '#D97706',
    'basin1'  : '#DC2626',
    'flow'    : '#7C3AED',
    'tide'    : '#0891B2',
    'sea'     : '#0891B2',
    'concrete': '#6B7280',
    'ship'    : '#374151',
    'bg'      : '#F8FAFC',
    'panel_bg': '#F1F5F9',
}

PHASE_COLORS = {
    WSB_DRAIN_1: '#DCFCE7',
    WSB_DRAIN_2: '#FEF9C3',
    WSB_DRAIN_3: '#FEE2E2',
    FINAL_DRAIN: '#EDE9FE',
}

PHASE_LABEL_COLORS = {
    WSB_DRAIN_1: '#059669',
    WSB_DRAIN_2: '#D97706',
    WSB_DRAIN_3: '#DC2626',
    FINAL_DRAIN: '#7C3AED',
}

PHASE_SHORT = {
    WSB_DRAIN_1: 'WSB DRAIN 1\nLock → Basin 3',
    WSB_DRAIN_2: 'WSB DRAIN 2\nLock → Basin 2',
    WSB_DRAIN_3: 'WSB DRAIN 3\nLock → Basin 1',
    FINAL_DRAIN: 'FINAL DRAIN\nLock → Sea',
}

COUPLING_LABEL = {
    WSB_DRAIN_1: 'Lock  ↔  Basin 3',
    WSB_DRAIN_2: 'Lock  ↔  Basin 2',
    WSB_DRAIN_3: 'Lock  ↔  Basin 1',
    FINAL_DRAIN: 'Lock  ↔  Sea',
}

# ── SimulationState ───────────────────────────────────────────────────────────

class SimulationState:
    """
    Holds the full simulation state and advances it one Euler step at a time.
    Designed for real-time animation — no solve_ivp, just direct ODE stepping.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.t        = 0.0
        self.phase    = 0
        self.t_entry  = 0.0
        self.done     = False
        self.y        = np.array([
            LOCK_H_UPPER,
            BASIN_3_H_INIT,
            BASIN_2_H_INIT,
            BASIN_1_H_INIT,
            0.0,
        ], dtype=float)

        # History buffers
        self.hist_t   = [0.0]
        self.hist_hL  = [LOCK_H_UPPER]
        self.hist_hB3 = [BASIN_3_H_INIT]
        self.hist_hB2 = [BASIN_2_H_INIT]
        self.hist_hB1 = [BASIN_1_H_INIT]
        self.hist_Q   = [0.0]

        # Phase transition log: list of (phase_idx, t_start, t_end or None)
        self.phase_log = [(0, 0.0, None)]

        # Flash counter for transition indicator (Task 4)
        self.flash_frames = 0

    def step(self, dt=1.0):
        """Advance simulation by dt seconds using Euler integration."""
        if self.done:
            return

        # Get derivatives from current phase ODE
        dydt = system_dynamics(self.t, self.y, self.phase, self.t_entry)
        self.y = self.y + np.array(dydt) * dt
        self.y[0] = max(self.y[0], 0.0)   # H_L >= 0
        self.y[4] = max(self.y[4], 0.0)   # Q >= 0
        self.t   += dt

        # Check transition condition
        target_idx, _ = PHASE_CONFIG[self.phase]
        h_target = H_sea(self.t) if target_idx == -1 else self.y[target_idx]
        delta = self.y[0] - h_target

        if delta <= P.EPSILON:
            self._transition()

        # Record history every 3 seconds of sim time
        if len(self.hist_t) == 0 or self.t - self.hist_t[-1] >= 3.0:
            self._record()

    def _transition(self):
        """Move to next FSM phase."""
        if self.phase >= len(STATE_NAMES) - 1:
            self.done = True
            self._record()
            # Close last phase log entry
            self.phase_log[-1] = (self.phase, self.phase_log[-1][1], self.t)
            return

        # Close current phase log entry
        self.phase_log[-1] = (self.phase, self.phase_log[-1][1], self.t)

        self.phase   += 1
        self.t_entry  = self.t
        self.y[4]     = 0.0   # reset Q at valve close
        self.flash_frames = 12  # Task 4: trigger flash

        # Open new phase log entry
        self.phase_log.append((self.phase, self.t, None))

    def _record(self):
        self.hist_t.append(self.t)
        self.hist_hL.append(self.y[0])
        self.hist_hB3.append(self.y[1])
        self.hist_hB2.append(self.y[2])
        self.hist_hB1.append(self.y[3])
        self.hist_Q.append(self.y[4])

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def hL(self):  return self.y[0]
    @property
    def hB3(self): return self.y[1]
    @property
    def hB2(self): return self.y[2]
    @property
    def hB1(self): return self.y[3]
    @property
    def Q(self):   return self.y[4]
    @property
    def phase_name(self): return STATE_NAMES[self.phase]


# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 9), facecolor=C['bg'])
fig.suptitle('', fontsize=1)  # placeholder — updated each frame

# Grid: top row = cross-section (left) + time series (right)
#       bottom row = sliders
gs = fig.add_gridspec(
    3, 2,
    height_ratios=[0.08, 0.82, 0.10],
    hspace=0.08,
    wspace=0.08,
    left=0.05, right=0.97,
    top=0.95, bottom=0.04,
)

ax_state  = fig.add_subplot(gs[0, :])   # FSM state indicator (Task 4)
ax_cross  = fig.add_subplot(gs[1, 0])   # cross-section view (Task 2)
ax_series = fig.add_subplot(gs[1, 1])   # time series (Task 3)
ax_slider = fig.add_subplot(gs[2, :])   # slider area (Task 5)

for ax in [ax_state, ax_slider]:
    ax.set_axis_off()

ax_cross.set_facecolor(C['panel_bg'])
ax_series.set_facecolor(C['panel_bg'])


# ── Simulation state & control flags ─────────────────────────────────────────
state   = SimulationState()
paused  = [False]
sim_speed = [30]   # sim seconds per animation frame


# ── Keyboard controls (Task 1) ────────────────────────────────────────────────
def on_key(event):
    if event.key == ' ':
        paused[0] = not paused[0]
    elif event.key == 'r':
        state.reset()
        paused[0] = False
    elif event.key == 'up':
        sim_speed[0] = min(sim_speed[0] + 10, 120)
        sl_speed.set_val(sim_speed[0])
    elif event.key == 'down':
        sim_speed[0] = max(sim_speed[0] - 10, 5)
        sl_speed.set_val(sim_speed[0])
    elif event.key in ('q', 'escape'):
        plt.close('all')

fig.canvas.mpl_connect('key_press_event', on_key)


# ── Slider widgets (Task 5) — defined early so keyboard handler can ref them ──
ax_sl_cd  = fig.add_axes([0.08, 0.025, 0.22, 0.018])
ax_sl_vt  = fig.add_axes([0.40, 0.025, 0.22, 0.018])
ax_sl_sp  = fig.add_axes([0.72, 0.025, 0.22, 0.018])
ax_btn    = fig.add_axes([0.46, 0.003, 0.08, 0.018])

sl_cd    = Slider(ax_sl_cd, 'Cd',         0.40, 1.00, valinit=P.CD,             valstep=0.01,  color='#93C5FD')
sl_vt    = Slider(ax_sl_vt, 'Valve (s)',  60,   600,  valinit=P.VALVE_OPEN_TIME, valstep=10,   color='#86EFAC')
sl_speed = Slider(ax_sl_sp, 'Speed ×',   5,    120,  valinit=sim_speed[0],      valstep=5,    color='#C4B5FD')
btn_reset = Button(ax_btn, 'Reset', color='#F1F5F9', hovercolor='#E2E8F0')

def on_cd_change(val):
    P.CD = val
def on_vt_change(val):
    P.VALVE_OPEN_TIME = val
def on_speed_change(val):
    sim_speed[0] = int(val)
def on_reset(event):
    state.reset()
    sl_cd.set_val(0.70);  P.CD = 0.70
    sl_vt.set_val(240);   P.VALVE_OPEN_TIME = 240
    sl_speed.set_val(30); sim_speed[0] = 30
    paused[0] = False

sl_cd.on_changed(on_cd_change)
sl_vt.on_changed(on_vt_change)
sl_speed.on_changed(on_speed_change)
btn_reset.on_clicked(on_reset)


# ── Task 4: FSM state indicator ───────────────────────────────────────────────
_state_text   = ax_state.text(0.5, 0.72, '', transform=ax_state.transAxes,
                               ha='center', va='center', fontsize=13, fontweight='bold')
_coupling_text = ax_state.text(0.5, 0.18, '', transform=ax_state.transAxes,
                                ha='center', va='center', fontsize=10, color='#64748B')
_time_text    = ax_state.text(0.02, 0.5, '', transform=ax_state.transAxes,
                               ha='left', va='center', fontsize=10, color='#64748B',
                               fontfamily='monospace')
_speed_text   = ax_state.text(0.98, 0.5, '', transform=ax_state.transAxes,
                               ha='right', va='center', fontsize=10, color='#64748B')

# Small static FSM chain diagram
_phase_boxes = []
for i, name in enumerate(STATE_NAMES):
    short = ['WSB 1', 'WSB 2', 'WSB 3', 'FINAL']
    x = 0.18 + i * 0.175
    box = ax_state.text(x, 0.5, short[i], transform=ax_state.transAxes,
                        ha='center', va='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor='#CBD5E1', linewidth=1.5),
                        color='#94A3B8')
    _phase_boxes.append(box)
    if i < len(STATE_NAMES) - 1:
        ax_state.annotate('', xy=(x + 0.14, 0.5), xytext=(x + 0.06, 0.5),
                          xycoords='axes fraction', textcoords='axes fraction',
                          arrowprops=dict(arrowstyle='->', color='#CBD5E1', lw=1.2))


def update_state_indicator(s):
    """Update FSM indicator panel. Called each frame."""
    phase_name  = s.phase_name
    lcolor      = PHASE_LABEL_COLORS[phase_name]
    flash       = s.flash_frames > 0

    bg = lcolor if flash else C['bg']
    fg = 'white' if flash else lcolor

    _state_text.set_text(PHASE_SHORT[phase_name])
    _state_text.set_color(fg)
    ax_state.set_facecolor(bg if flash else C['bg'])

    _coupling_text.set_text(COUPLING_LABEL[phase_name])

    mins = int(s.t // 60); secs = int(s.t % 60)
    status = 'PAUSED' if paused[0] else ('DONE' if s.done else 'RUNNING')
    _time_text.set_text(f't = {mins:02d}:{secs:02d}   [{status}]')
    _speed_text.set_text(f'Speed: {sim_speed[0]}×  |  SPACE=pause  R=reset  ↑↓=speed')

    # Highlight current phase box
    for i, box in enumerate(_phase_boxes):
        active = (i == s.phase)
        box.set_color(PHASE_LABEL_COLORS[STATE_NAMES[i]] if active else '#94A3B8')
        box.get_bbox_patch().set_edgecolor(
            PHASE_LABEL_COLORS[STATE_NAMES[i]] if active else '#CBD5E1')
        box.get_bbox_patch().set_linewidth(2.5 if active else 1.5)
        box.get_bbox_patch().set_facecolor(
            PHASE_COLORS[STATE_NAMES[i]] if active else 'white')

    if s.flash_frames > 0:
        s.flash_frames -= 1


# ── Task 2: Cross-section drawing ────────────────────────────────────────────
def draw_cross_section(ax, s):
    ax.cla()
    ax.set_facecolor(C['panel_bg'])
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 30)
    ax.set_title('Lock Cross-Section', fontsize=11, fontweight='bold',
                 color='#1E293B', pad=6)
    ax.set_ylabel('Water level (m)', fontsize=9, color='#64748B')
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=8, colors='#64748B')
    ax.spines[['top','right','bottom']].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')

    MAX_H    = 28.0
    WALL_T   = 0.15

    # Structure layout: [x_left, width, label, color, current_h]
    structures = [
        (0.2,  3.5, 'LOCK',    C['lock'],   s.hL),
        (4.2,  1.6, 'Basin 3', C['basin3'], s.hB3),
        (6.1,  1.6, 'Basin 2', C['basin2'], s.hB2),
        (8.0,  1.6, 'Basin 1', C['basin1'], s.hB1),
    ]

    # Sea on the left (narrow)
    sea_h = H_sea(s.t) + 1.5  # offset so it's always a bit visible
    ax.add_patch(Rectangle((-0.05, 0), 0.18, sea_h,
                            facecolor=C['sea'], alpha=0.4, zorder=2))
    ax.add_patch(Rectangle((-0.05, 0), 0.18, MAX_H,
                            facecolor='none', edgecolor=C['concrete'],
                            linewidth=1.5, zorder=3))
    ax.text(0.09, sea_h + 0.4, f'{H_sea(s.t):.1f}m', ha='center',
            fontsize=6.5, color=C['sea'], zorder=5)
    ax.text(0.09, -0.7, 'SEA', ha='center', fontsize=6, color=C['sea'])

    active_pair = PHASE_CONFIG[s.phase]

    for idx, (xl, w, label, color, h) in enumerate(structures):
        h = max(h, 0)

        # Determine if this structure is actively coupled
        # idx 0=lock, 1=B3, 2=B2, 3=B1
        # active pair target_idx: 1=B3,2=B2,3=B1,-1=sea
        target_idx = active_pair[0]
        is_active = (idx == 0) or \
                    (idx == 1 and target_idx == 1) or \
                    (idx == 2 and target_idx == 2) or \
                    (idx == 3 and target_idx == 3) or \
                    (idx == 0 and target_idx == -1)

        # Concrete walls
        edge_lw  = 2.5 if is_active else 1.2
        edge_col = color if is_active else C['concrete']
        ax.add_patch(Rectangle((xl, 0), w, MAX_H,
                               facecolor='#F8FAFC', edgecolor=edge_col,
                               linewidth=edge_lw, zorder=2))

        # Water fill
        ax.add_patch(Rectangle((xl + WALL_T, 0), w - 2*WALL_T, h,
                               facecolor=color, alpha=0.55, zorder=3))

        # Water surface line
        ax.plot([xl + WALL_T, xl + w - WALL_T], [h, h],
                color=color, lw=2.0, zorder=4)

        # Level label
        ax.text(xl + w/2, h + 0.5, f'{h:.1f}m',
                ha='center', fontsize=7.5, color=color,
                fontweight='bold', zorder=5)

        # Structure label
        ax.text(xl + w/2, -0.75, label,
                ha='center', fontsize=7, color=color, fontweight='bold')

    # Ship in lock
    lock_x = structures[0][0]; lock_w = structures[0][1]
    ship_y = max(s.hL - 1.5, 0.2)
    ship_pts = np.array([
        [lock_x + 0.35, ship_y],
        [lock_x + 0.55, ship_y + 1.4],
        [lock_x + lock_w - 0.55, ship_y + 1.4],
        [lock_x + lock_w - 0.35, ship_y],
    ])
    ax.add_patch(Polygon(ship_pts, closed=True,
                         facecolor=C['ship'], alpha=0.85, zorder=5))
    # Superstructure
    ax.add_patch(Rectangle((lock_x + 1.1, ship_y + 1.4), 0.9, 0.9,
                            facecolor='#4B5563', zorder=6))
    ax.add_patch(Rectangle((lock_x + 1.5, ship_y + 2.3), 0.3, 0.6,
                            facecolor='#6B7280', zorder=7))

    # Flow arrow between active pair
    Q_norm = min(s.Q / 800, 1.0)
    if Q_norm > 0.02 and not s.done:
        target_idx = active_pair[0]
        if target_idx == -1:
            # Lock → Sea
            x1 = structures[0][0] + WALL_T
            x2 = 0.18
            y_arr = max(s.hL * 0.45, 2.0)
            ax.annotate('', xy=(x2, y_arr), xytext=(x1, y_arr),
                        arrowprops=dict(
                            arrowstyle='->', color=C['sea'],
                            lw=1.5 + Q_norm * 2,
                            connectionstyle='arc3,rad=0.1'))
        else:
            # Lock → Basin
            basin_map = {1: 1, 2: 2, 3: 3}
            b_idx = basin_map[target_idx]
            x1 = structures[0][0] + structures[0][1] - WALL_T
            x2 = structures[b_idx][0] + WALL_T
            y_arr = max(min(s.hL, structures[b_idx][4]) * 0.6, 2.0)
            arrow_color = [C['basin3'], C['basin2'], C['basin1']][b_idx - 1]
            ax.annotate('', xy=(x2, y_arr), xytext=(x1, y_arr),
                        arrowprops=dict(
                            arrowstyle='->', color=arrow_color,
                            lw=1.5 + Q_norm * 2.5,
                            connectionstyle='arc3,rad=-0.15'))

        # Q label on arrow
        mid_x = (x1 + x2) / 2 if target_idx != -1 else (x1 + x2) / 2
        ax.text(mid_x, y_arr + 0.7, f'Q={s.Q:.0f} m³/s',
                ha='center', fontsize=7.5, color='#374151',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                zorder=8)

    # Valve ramp indicator
    Av_frac = valve_area(s.t, s.t_entry) / P.A_VALVE_MAX
    ax.text(9.85, 1.0, f'Valve\n{Av_frac*100:.0f}%',
            ha='right', va='bottom', fontsize=7, color='#64748B')
    ax.add_patch(Rectangle((9.5, 0.2), 0.3, 2.5,
                            facecolor='#E2E8F0', edgecolor='#CBD5E1', lw=1))
    ax.add_patch(Rectangle((9.5, 0.2), 0.3, 2.5 * Av_frac,
                            facecolor='#7C3AED', alpha=0.7))

    # Done overlay
    if s.done:
        ax.text(5.0, 14, 'CYCLE\nCOMPLETE', ha='center', va='center',
                fontsize=18, fontweight='bold', color='#059669', alpha=0.35,
                rotation=15, zorder=10)


# ── Task 3: Time series drawing ───────────────────────────────────────────────
def draw_time_series(ax, s):
    ax.cla()
    ax.set_facecolor(C['panel_bg'])
    ax.set_title('Water Levels & Flow Rate', fontsize=11, fontweight='bold',
                 color='#1E293B', pad=6)
    ax.set_ylabel('Water level (m)', fontsize=9, color='#64748B')
    ax.set_xlabel('Time (min)', fontsize=9, color='#64748B')
    ax.tick_params(labelsize=8, colors='#64748B')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#CBD5E1')

    t_arr   = np.array(s.hist_t)   / 60
    hL_arr  = np.array(s.hist_hL)
    hB3_arr = np.array(s.hist_hB3)
    hB2_arr = np.array(s.hist_hB2)
    hB1_arr = np.array(s.hist_hB1)
    Q_arr   = np.array(s.hist_Q)

    if len(t_arr) < 2:
        ax.set_xlim(0, 25)
        ax.set_ylim(-1, 30)
        return

    # Scrolling window: show last 25 min
    t_now = s.t / 60
    x_max = max(t_now + 1, 25)
    x_min = max(0, x_max - 25)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, 30)

    # Phase background shading
    for phase_idx, t_start, t_end in s.phase_log:
        t0 = t_start / 60
        t1 = (t_end if t_end is not None else s.t) / 60
        ax.axvspan(t0, t1, alpha=0.25,
                   color=PHASE_COLORS[STATE_NAMES[phase_idx]], zorder=0)
        # Phase label at top
        if t1 - t0 > 0.5:
            ax.text((t0 + t1) / 2, 28.2,
                    ['WSB1','WSB2','WSB3','FINAL'][phase_idx],
                    ha='center', fontsize=7,
                    color=PHASE_LABEL_COLORS[STATE_NAMES[phase_idx]])

    # Water level traces
    ax.plot(t_arr, hL_arr,  color=C['lock'],   lw=2.0, label='Lock H_L', zorder=3)
    ax.plot(t_arr, hB3_arr, color=C['basin3'], lw=1.5, label='Basin 3',  zorder=3, ls='--')
    ax.plot(t_arr, hB2_arr, color=C['basin2'], lw=1.5, label='Basin 2',  zorder=3, ls='--')
    ax.plot(t_arr, hB1_arr, color=C['basin1'], lw=1.5, label='Basin 1',  zorder=3, ls='--')

    # Q on secondary axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Flow rate Q (m³/s)', fontsize=9, color=C['flow'])
    ax2.tick_params(labelsize=8, colors=C['flow'])
    ax2.spines[['top','bottom']].set_visible(False)
    ax2.spines['right'].set_color(C['flow'])
    ax2.plot(t_arr, Q_arr, color=C['flow'], lw=1.5, ls=':', label='Q', zorder=2, alpha=0.8)
    ax2.set_ylim(0, max(Q_arr.max() * 1.3, 100))

    # Current time line
    ax.axvline(t_now, color='#94A3B8', lw=1.0, ls='--', zorder=4)

    # Legend
    handles = [
        mpatches.Patch(color=C['lock'],   label='Lock H_L'),
        mpatches.Patch(color=C['basin3'], label='Basin 3'),
        mpatches.Patch(color=C['basin2'], label='Basin 2'),
        mpatches.Patch(color=C['basin1'], label='Basin 1'),
        mpatches.Patch(color=C['flow'],   label='Q (m³/s)'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7.5,
              framealpha=0.85, edgecolor='#E2E8F0')

    # Water recovery annotation
    if s.t > 30:
        dB3 = max(s.hB3 - BASIN_3_H_INIT, 0)
        dB2 = max(s.hB2 - BASIN_2_H_INIT, 0)
        dB1 = max(s.hB1 - BASIN_1_H_INIT, 0)
        from parameters import BASIN_AREA, LOCK_AREA
        vol_rec  = (dB3 + dB2 + dB1) * BASIN_AREA
        vol_tot  = max(LOCK_H_UPPER - s.hL, 0.01) * LOCK_AREA
        pct      = min(vol_rec / vol_tot * 100, 100)
        ax.text(x_min + 0.3, 1.5, f'Recovery: {pct:.1f}%',
                fontsize=8, color='#059669',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='#059669',
                          boxstyle='round,pad=0.3'))


# ── Animation update function ─────────────────────────────────────────────────
def update(frame):
    # Advance simulation
    if not paused[0] and not state.done:
        dt = 1.0
        for _ in range(sim_speed[0]):
            if not state.done:
                state.step(dt)

    # Redraw panels
    draw_cross_section(ax_cross, state)
    draw_time_series(ax_series, state)
    update_state_indicator(state)

    return []


# ── Run ───────────────────────────────────────────────────────────────────────
ani = animation.FuncAnimation(
    fig,
    update,
    interval=33,     # ~30 fps
    blit=False,
    cache_frame_data=False,
)

plt.show()