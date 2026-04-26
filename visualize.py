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
from matplotlib.patches import Rectangle, Polygon

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
    WSB_DRAIN_1: 'WSB DRAIN 1  —  Lock → Basin 3',
    WSB_DRAIN_2: 'WSB DRAIN 2  —  Lock → Basin 2',
    WSB_DRAIN_3: 'WSB DRAIN 3  —  Lock → Basin 1',
    FINAL_DRAIN: 'FINAL DRAIN  —  Lock → Sea',
}
PHASE_ABBR     = ['WSB 1', 'WSB 2', 'WSB 3', 'FINAL']
COUPLING_LABEL = ['Lock ↔ Basin 3','Lock ↔ Basin 2','Lock ↔ Basin 1','Lock ↔ Sea']

# ── SimulationState ───────────────────────────────────────────────────────────
class SimulationState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t           = 0.0
        self.phase       = 0
        self.t_entry     = 0.0
        self.done        = False
        self.y           = np.array([LOCK_H_UPPER, BASIN_3_H_INIT,
                                      BASIN_2_H_INIT, BASIN_1_H_INIT, 0.0], dtype=float)
        self.hist_t      = [0.0]
        self.hist_hL     = [LOCK_H_UPPER]
        self.hist_hB3    = [BASIN_3_H_INIT]
        self.hist_hB2    = [BASIN_2_H_INIT]
        self.hist_hB1    = [BASIN_1_H_INIT]
        self.hist_Q      = [0.0]
        self.phase_log   = [(0, 0.0, None)]
        self.flash_frames = 0
        self._last_record = 0.0

    def step(self, dt=1.0):
        if self.done:
            return
        dydt      = system_dynamics(self.t, self.y, self.phase, self.t_entry)
        self.y    = self.y + np.array(dydt) * dt
        self.y[0] = max(self.y[0], 0.0)
        self.y[4] = max(self.y[4], 0.0)
        self.t   += dt

        target_idx, _ = PHASE_CONFIG[self.phase]
        h_target = H_sea(self.t) if target_idx == -1 else self.y[target_idx]
        if self.y[0] - h_target <= P.EPSILON:
            self._transition()

        if self.t - self._last_record >= 3.0:
            self._record()
            self._last_record = self.t

    def _transition(self):
        self.phase_log[-1] = (self.phase, self.phase_log[-1][1], self.t)
        if self.phase >= len(STATE_NAMES) - 1:
            self.done = True
            self._record()
            return
        self.phase       += 1
        self.t_entry      = self.t
        self.y[4]         = 0.0
        self.flash_frames = 15
        self.phase_log.append((self.phase, self.t, None))

    def _record(self):
        self.hist_t.append(self.t)
        self.hist_hL.append(self.y[0])
        self.hist_hB3.append(self.y[1])
        self.hist_hB2.append(self.y[2])
        self.hist_hB1.append(self.y[3])
        self.hist_Q.append(self.y[4])

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
gs  = fig.add_gridspec(
    3, 2,
    height_ratios=[0.07, 0.83, 0.10],
    hspace=0.06, wspace=0.10,
    left=0.05, right=0.97, top=0.96, bottom=0.04,
)
ax_state  = fig.add_subplot(gs[0, :])
ax_cross  = fig.add_subplot(gs[1, 0])
ax_series = fig.add_subplot(gs[1, 1])
ax_slider = fig.add_subplot(gs[2, :])

for ax in [ax_state, ax_slider]:
    ax.set_axis_off()
ax_cross.set_facecolor(C['panel_bg'])
ax_series.set_facecolor(C['panel_bg'])

state     = SimulationState()
paused    = [False]
sim_speed = [5]   # default 5× — slow enough to watch the physics


# ── Sliders (Task 5) ──────────────────────────────────────────────────────────
ax_sl_cd  = fig.add_axes([0.08, 0.025, 0.22, 0.018])
ax_sl_vt  = fig.add_axes([0.40, 0.025, 0.22, 0.018])
ax_sl_sp  = fig.add_axes([0.72, 0.025, 0.22, 0.018])
ax_btn    = fig.add_axes([0.455, 0.003, 0.09, 0.018])

sl_cd    = Slider(ax_sl_cd, 'Cd',        0.40, 1.00, valinit=P.CD,              valstep=0.01, color='#93C5FD')
sl_vt    = Slider(ax_sl_vt, 'Valve (s)', 60,   600,  valinit=P.VALVE_OPEN_TIME, valstep=10,  color='#86EFAC')
sl_speed = Slider(ax_sl_sp, 'Speed ×',  1,    60,   valinit=sim_speed[0],      valstep=1,   color='#C4B5FD')
btn_reset = Button(ax_btn, 'Reset', color='#E2E8F0', hovercolor='#CBD5E1')

sl_cd.on_changed(lambda v: setattr(P, 'CD', v))
sl_vt.on_changed(lambda v: setattr(P, 'VALVE_OPEN_TIME', v))
sl_speed.on_changed(lambda v: sim_speed.__setitem__(0, int(v)))

def on_reset(event):
    state.reset()
    sl_cd.set_val(0.70);  P.CD = 0.70
    sl_vt.set_val(240);   P.VALVE_OPEN_TIME = 240
    sl_speed.set_val(5);  sim_speed[0] = 5
    paused[0] = False
    # Wipe phase spans
    _series_artists['prev_n_phases'] = -1

btn_reset.on_clicked(on_reset)


# ── Keyboard (Task 1) ─────────────────────────────────────────────────────────
def on_key(event):
    if event.key == ' ':
        paused[0] = not paused[0]
    elif event.key == 'r':
        on_reset(None)
    elif event.key == 'up':
        sim_speed[0] = min(sim_speed[0] + 1, 60)
        sl_speed.set_val(sim_speed[0])
    elif event.key == 'down':
        sim_speed[0] = max(sim_speed[0] - 1, 1)
        sl_speed.set_val(sim_speed[0])
    elif event.key in ('q', 'escape'):
        plt.close('all')

fig.canvas.mpl_connect('key_press_event', on_key)


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-SECTION PANEL (Task 2) — artists created once, updated each frame
# ═══════════════════════════════════════════════════════════════════════════════
MAX_H  = 28.0
WALL_T = 0.15

# [x_left, width, color, label]
STRUCTS = [
    (0.20, 3.50, C['lock'],   'LOCK'),
    (4.20, 1.60, C['basin3'], 'Basin 3'),
    (6.10, 1.60, C['basin2'], 'Basin 2'),
    (8.00, 1.60, C['basin1'], 'Basin 1'),
]

# Static setup
ax_cross.set_xlim(0, 10)
ax_cross.set_ylim(-1.2, MAX_H + 2)
ax_cross.set_xticks([])
ax_cross.tick_params(axis='y', labelsize=8, colors='#64748B')
ax_cross.spines[['top','right','bottom']].set_visible(False)
ax_cross.spines['left'].set_color('#CBD5E1')
ax_cross.set_ylabel('Water level (m)', fontsize=9, color='#64748B')
ax_cross.set_title('Lock Cross-Section', fontsize=11,
                    fontweight='bold', color='#1E293B', pad=6)

# Static concrete outlines
for xl, w, color, lbl in STRUCTS:
    ax_cross.add_patch(Rectangle((xl, 0), w, MAX_H,
                                  facecolor='#F8FAFC', edgecolor=C['concrete'],
                                  linewidth=1.2, zorder=2))
    ax_cross.text(xl + w/2, -0.9, lbl, ha='center', fontsize=6.5,
                  color=color, fontweight='bold')
ax_cross.add_patch(Rectangle((-0.05, 0), 0.18, MAX_H,
                               facecolor='none', edgecolor=C['concrete'],
                               linewidth=1.2, zorder=2))
ax_cross.text(0.09, -0.9, 'SEA', ha='center', fontsize=6,
              color=C['sea'], fontweight='bold')

# Mutable cross-section artists
_ca = {}
_ca['sea_fill'] = ax_cross.add_patch(
    Rectangle((0.09, 0), 0.09, 0, facecolor=C['sea'], alpha=0.45, zorder=3))
_ca['sea_surf'], = ax_cross.plot([], [], color=C['sea'], lw=2.0, zorder=4)
_ca['sea_lbl']  = ax_cross.text(0.09, 0.3, '', ha='center',
                                 fontsize=6.5, color=C['sea'], zorder=5)

for i, (xl, w, color, _) in enumerate(STRUCTS):
    _ca[f'fill_{i}']   = ax_cross.add_patch(
        Rectangle((xl + WALL_T, 0), w - 2*WALL_T, 0,
                   facecolor=color, alpha=0.55, zorder=3))
    _ca[f'surf_{i}'], = ax_cross.plot([], [], color=color, lw=2.0, zorder=4)
    _ca[f'lbl_{i}']   = ax_cross.text(xl + w/2, 0.3, '', ha='center',
                                       fontsize=7.5, color=color,
                                       fontweight='bold', zorder=5)
    _ca[f'border_{i}'] = ax_cross.add_patch(
        Rectangle((xl, 0), w, MAX_H, facecolor='none',
                   edgecolor=color, linewidth=2.5, zorder=5, alpha=0))

_ca['ship_hull'] = ax_cross.add_patch(
    Polygon(np.zeros((4,2)), closed=True, facecolor=C['ship'], alpha=0.85, zorder=6))
_ca['ship_cab']  = ax_cross.add_patch(
    Rectangle((0,0), 0.9, 0.9, facecolor='#4B5563', zorder=7))
_ca['ship_mast'] = ax_cross.add_patch(
    Rectangle((0,0), 0.3, 0.6, facecolor='#6B7280', zorder=8))

_ca['arrow'],    = ax_cross.plot([], [], color=C['basin3'], lw=3, zorder=7, alpha=0,
                                  marker='>', markersize=7, markevery=[-1])
_ca['arrow_txt'] = ax_cross.text(5, 8, '', ha='center', fontsize=7.5,
                                  color='#374151', zorder=9,
                                  bbox=dict(facecolor='white', alpha=0,
                                            edgecolor='none', pad=1))

_ca['valve_bg']   = ax_cross.add_patch(
    Rectangle((9.5, 0.2), 0.3, 2.5, facecolor='#E2E8F0', edgecolor='#CBD5E1', lw=1, zorder=4))
_ca['valve_fill'] = ax_cross.add_patch(
    Rectangle((9.5, 0.2), 0.3, 0, facecolor='#7C3AED', alpha=0.75, zorder=5))
_ca['valve_txt']  = ax_cross.text(9.85, 0.8, 'Valve\n0%',
                                   ha='right', va='bottom', fontsize=7,
                                   color='#64748B', zorder=6)
_ca['done_txt']   = ax_cross.text(5.0, 14, '', ha='center', va='center',
                                   fontsize=20, fontweight='bold',
                                   color='#059669', alpha=0, rotation=15, zorder=10)


def _update_cross(s):
    # Sea
    sea_h = max(H_sea(s.t) + 1.5, 0.05)
    _ca['sea_fill'].set_height(sea_h)
    _ca['sea_surf'].set_data([0.09, 0.17], [sea_h, sea_h])
    _ca['sea_lbl'].set_position((0.09, sea_h + 0.25))
    _ca['sea_lbl'].set_text(f'{H_sea(s.t):.1f}m')

    heights     = [s.hL, s.hB3, s.hB2, s.hB1]
    target_idx  = PHASE_CONFIG[s.phase][0]
    # target_idx 1,2,3 → struct index 1,2,3; -1 (sea) → no basin active
    active_struct = target_idx if target_idx in (1,2,3) else -1

    for i, (xl, w, color, _) in enumerate(STRUCTS):
        h = max(heights[i], 0)
        _ca[f'fill_{i}'].set_height(h)
        _ca[f'surf_{i}'].set_data([xl + WALL_T, xl + w - WALL_T], [h, h])
        _ca[f'lbl_{i}'].set_position((xl + w/2, h + 0.3))
        _ca[f'lbl_{i}'].set_text(f'{h:.1f}m')
        is_active = (i == 0) or (i == active_struct)
        _ca[f'border_{i}'].set_alpha(0.9 if is_active else 0)

    # Ship
    lx, lw = STRUCTS[0][0], STRUCTS[0][1]
    sy = max(s.hL - 1.5, 0.2)
    _ca['ship_hull'].set_xy(np.array([
        [lx + 0.35, sy],
        [lx + 0.55, sy + 1.4],
        [lx + lw - 0.55, sy + 1.4],
        [lx + lw - 0.35, sy],
    ]))
    _ca['ship_cab'].set_xy((lx + 1.1, sy + 1.4))
    _ca['ship_mast'].set_xy((lx + 1.5, sy + 2.3))

    # Flow arrow
    Q_norm = min(s.Q / 600, 1.0)
    if Q_norm > 0.03 and not s.done:
        if target_idx == -1:
            x1, x2    = STRUCTS[0][0] + WALL_T, 0.17
            acol      = C['sea']
        else:
            x1        = STRUCTS[0][0] + STRUCTS[0][1] - WALL_T
            x2        = STRUCTS[active_struct][0] + WALL_T
            acol      = STRUCTS[active_struct][2]
        y_a = max(min(s.hL, heights[active_struct] if active_struct >= 0 else s.hL) * 0.5, 2.5)
        _ca['arrow'].set_data([x1, x2], [y_a, y_a])
        _ca['arrow'].set_color(acol)
        _ca['arrow'].set_linewidth(1.5 + Q_norm * 3)
        _ca['arrow'].set_alpha(0.4 + Q_norm * 0.6)
        _ca['arrow_txt'].set_position(((x1+x2)/2, y_a + 0.55))
        _ca['arrow_txt'].set_text(f'Q = {s.Q:.0f} m³/s')
        _ca['arrow_txt'].get_bbox_patch().set_alpha(0.75)
    else:
        _ca['arrow'].set_alpha(0)
        _ca['arrow_txt'].set_text('')
        if _ca['arrow_txt'].get_bbox_patch():
            _ca['arrow_txt'].get_bbox_patch().set_alpha(0)

    # Valve indicator
    Av_frac = min(valve_area(s.t, s.t_entry) / max(P.A_VALVE_MAX, 1e-6), 1.0)
    _ca['valve_fill'].set_height(2.5 * Av_frac)
    _ca['valve_txt'].set_text(f'Valve\n{Av_frac*100:.0f}%')

    # Done overlay
    _ca['done_txt'].set_text('CYCLE\nCOMPLETE' if s.done else '')
    _ca['done_txt'].set_alpha(0.35 if s.done else 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TIME SERIES PANEL (Task 3) — artist-update pattern
# ═══════════════════════════════════════════════════════════════════════════════
ax_series.set_xlim(0, 25)
ax_series.set_ylim(-1, 30)
ax_series.set_xlabel('Time (min)', fontsize=9, color='#64748B')
ax_series.set_ylabel('Water level (m)', fontsize=9, color='#64748B')
ax_series.set_title('Water Levels & Flow Rate', fontsize=11,
                     fontweight='bold', color='#1E293B', pad=6)
ax_series.tick_params(labelsize=8, colors='#64748B')
ax_series.spines[['top','right']].set_visible(False)
ax_series.spines[['left','bottom']].set_color('#CBD5E1')

ax_s2 = ax_series.twinx()
ax_s2.set_ylabel('Q (m³/s)', fontsize=9, color=C['flow'])
ax_s2.tick_params(labelsize=8, colors=C['flow'])
ax_s2.spines[['top','bottom']].set_visible(False)
ax_s2.spines['right'].set_color(C['flow'])
ax_s2.set_ylim(0, 2000)

_sa = {}
_sa['hL'],  = ax_series.plot([], [], color=C['lock'],   lw=2.0, zorder=3)
_sa['hB3'], = ax_series.plot([], [], color=C['basin3'], lw=1.5, ls='--', zorder=3)
_sa['hB2'], = ax_series.plot([], [], color=C['basin2'], lw=1.5, ls='--', zorder=3)
_sa['hB1'], = ax_series.plot([], [], color=C['basin1'], lw=1.5, ls='--', zorder=3)
_sa['Q'],   = ax_s2.plot([], [], color=C['flow'], lw=1.5, ls=':', alpha=0.85, zorder=2)
_sa['vline'],= ax_series.plot([], [], color='#94A3B8', lw=1.0, ls='--', zorder=4)
_sa['rec']   = ax_series.text(0.5, 1.5, '', fontsize=8, color='#059669', zorder=5,
                                bbox=dict(facecolor='white', alpha=0,
                                          edgecolor='#059669', boxstyle='round,pad=0.3'))
_sa['spans']        = []
_sa['span_labels']  = []
_sa['prev_n_phases'] = 0

# Static legend
ax_series.legend(handles=[
    mpatches.Patch(color=C['lock'],   label='Lock H_L'),
    mpatches.Patch(color=C['basin3'], label='Basin 3'),
    mpatches.Patch(color=C['basin2'], label='Basin 2'),
    mpatches.Patch(color=C['basin1'], label='Basin 1'),
    mpatches.Patch(color=C['flow'],   label='Q (m³/s)'),
], loc='upper right', fontsize=7.5, framealpha=0.85, edgecolor='#E2E8F0')


def _update_series(s):
    t_arr  = np.array(s.hist_t)  / 60
    hL_arr = np.array(s.hist_hL)
    hB3    = np.array(s.hist_hB3)
    hB2    = np.array(s.hist_hB2)
    hB1    = np.array(s.hist_hB1)
    Q_arr  = np.array(s.hist_Q)

    if len(t_arr) < 2:
        return

    t_now = s.t / 60
    x_max = max(t_now + 0.5, 25)
    x_min = max(0.0, x_max - 25)
    ax_series.set_xlim(x_min, x_max)

    _sa['hL'].set_data(t_arr, hL_arr)
    _sa['hB3'].set_data(t_arr, hB3)
    _sa['hB2'].set_data(t_arr, hB2)
    _sa['hB1'].set_data(t_arr, hB1)
    _sa['Q'].set_data(t_arr, Q_arr)
    ax_s2.set_ylim(0, max(Q_arr.max() * 1.3, 200))
    _sa['vline'].set_data([t_now, t_now], [-1, 30])

    # Phase spans — only rebuild on phase count change
    n = len(s.phase_log)
    if n != _sa['prev_n_phases']:
        for obj in _sa['spans'] + _sa['span_labels']:
            try: obj.remove()
            except: pass
        _sa['spans']       = []
        _sa['span_labels'] = []
        for pi, t0s, t1s in s.phase_log:
            t0 = t0s / 60
            t1 = (t1s if t1s is not None else s.t) / 60
            span = ax_series.axvspan(t0, t1, alpha=0.22,
                                      color=PHASE_COLORS[STATE_NAMES[pi]], zorder=0)
            _sa['spans'].append(span)
            if t1 - t0 > 0.3:
                lbl = ax_series.text((t0+t1)/2, 28.5, PHASE_ABBR[pi],
                                      ha='center', fontsize=6.5,
                                      color=PHASE_LABEL_COLORS[STATE_NAMES[pi]], zorder=1)
                _sa['span_labels'].append(lbl)
        _sa['prev_n_phases'] = n
    elif _sa['spans']:
        # Extend last open span
        pi, t0s, t1s = s.phase_log[-1]
        if t1s is None:
            t0 = t0s / 60
            t1 = s.t  / 60
            try:
                _sa['spans'][-1].set_xy([[t0,0],[t0,1],[t1,1],[t1,0]])
            except Exception:
                pass

    # Recovery annotation
    dB3 = max(s.hB3 - BASIN_3_H_INIT, 0)
    dB2 = max(s.hB2 - BASIN_2_H_INIT, 0)
    dB1 = max(s.hB1 - BASIN_1_H_INIT, 0)
    vol_rec = (dB3 + dB2 + dB1) * P.BASIN_AREA
    vol_tot = max(LOCK_H_UPPER - s.hL, 0.01) * P.LOCK_AREA
    pct = min(vol_rec / vol_tot * 100, 100)
    if s.t > 60:
        _sa['rec'].set_text(f'Recovery: {pct:.1f}%')
        _sa['rec'].set_position((x_min + 0.4, 1.5))
        _sa['rec'].get_bbox_patch().set_alpha(0.8)
    else:
        _sa['rec'].get_bbox_patch().set_alpha(0)


# ═══════════════════════════════════════════════════════════════════════════════
# FSM STATE INDICATOR (Task 4)
# ═══════════════════════════════════════════════════════════════════════════════
_st_main = ax_state.text(0.50, 0.75, '', transform=ax_state.transAxes,
                          ha='center', va='center', fontsize=12, fontweight='bold')
_st_coup = ax_state.text(0.50, 0.18, '', transform=ax_state.transAxes,
                          ha='center', va='center', fontsize=9, color='#64748B')
_st_time = ax_state.text(0.02, 0.50, '', transform=ax_state.transAxes,
                          ha='left', va='center', fontsize=9,
                          color='#64748B', fontfamily='monospace')
_st_spd  = ax_state.text(0.98, 0.50, '', transform=ax_state.transAxes,
                          ha='right', va='center', fontsize=9, color='#64748B')

_fsm_boxes = []
for i in range(len(STATE_NAMES)):
    x = 0.18 + i * 0.175
    bx = ax_state.text(x, 0.50, PHASE_ABBR[i], transform=ax_state.transAxes,
                       ha='center', va='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                 edgecolor='#CBD5E1', linewidth=1.5),
                       color='#94A3B8')
    _fsm_boxes.append(bx)
    if i < len(STATE_NAMES) - 1:
        ax_state.annotate('', xy=(x + 0.145, 0.50), xytext=(x + 0.055, 0.50),
                          xycoords='axes fraction', textcoords='axes fraction',
                          arrowprops=dict(arrowstyle='->', color='#CBD5E1', lw=1.2))


def _update_state(s):
    pname  = s.phase_name
    lcolor = PHASE_LABEL_COLORS[pname]
    flash  = s.flash_frames > 0

    _st_main.set_text(PHASE_SHORT[pname])
    _st_main.set_color('white' if flash else lcolor)
    ax_state.set_facecolor(lcolor if flash else C['bg'])

    _st_coup.set_text(f'Active coupling:  {COUPLING_LABEL[s.phase]}')

    m = int(s.t // 60);  sc = int(s.t % 60)
    status = 'PAUSED' if paused[0] else ('DONE ✓' if s.done else 'RUNNING')
    _st_time.set_text(f't = {m:02d}:{sc:02d}   [ {status} ]')
    _st_spd.set_text(f'Speed: {sim_speed[0]}×   |   SPACE = pause   R = reset   ↑↓ = speed')

    for i, bx in enumerate(_fsm_boxes):
        active = (i == s.phase)
        bx.set_color(PHASE_LABEL_COLORS[STATE_NAMES[i]] if active else '#94A3B8')
        bx.get_bbox_patch().set_edgecolor(
            PHASE_LABEL_COLORS[STATE_NAMES[i]] if active else '#CBD5E1')
        bx.get_bbox_patch().set_linewidth(2.5 if active else 1.5)
        bx.get_bbox_patch().set_facecolor(
            PHASE_COLORS[STATE_NAMES[i]] if active else 'white')

    if s.flash_frames > 0:
        s.flash_frames -= 1


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATION LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def update(frame):
    if not paused[0] and not state.done:
        for _ in range(sim_speed[0]):
            if not state.done:
                state.step(1.0)

    _update_cross(state)
    _update_series(state)
    _update_state(state)
    return []


ani = animation.FuncAnimation(
    fig, update,
    interval=40,           # 25 fps
    blit=False,
    cache_frame_data=False,
)

plt.show()