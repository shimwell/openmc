"""Midplane mesh-tally visualizations for the labyrinth A/B/C comparison.

Produces three figures in omega_bench/plots/:
  lab_flux_maps.png    log10 flux midplane, analog vs fw_cadis vs omega
  lab_relerr_maps.png  relative error midplane, same three cases
  lab_err_ratio.png    diverging map of rel-err ratio omega/fw_cadis
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openmc
from matplotlib.colors import LogNorm, TwoSlopeNorm

BASE = '/home/jon/openmc/omega_bench'
OUT = f'{BASE}/plots'
os.makedirs(OUT, exist_ok=True)

NX, NY, NZ = 60, 50, 23
EXTENT = [-100, 1100, -100, 900]  # x, y in cm
K_MID = 11  # z = 120..140 cm slab (corridor midheight)

INK = '#333333'
MUTED = '#777777'
plt.rcParams.update({
    'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': '#cccccc', 'font.size': 10,
})

CASES = [
    ('analog', f'{BASE}/lab_a2/statepoint.30.h5', f'{BASE}/lab_a2/result_analog_s2.json'),
    ('fw_cadis', f'{BASE}/lab_f3/statepoint.12.h5', f'{BASE}/lab_f3/result_fw_cadis_s3.json'),
    ('fw_cadis_omega', f'{BASE}/lab_o3/statepoint.12.h5', f'{BASE}/lab_o3/result_fw_cadis_omega_s3.json'),
]

# Corridor outline segments (x0,y0,x1,y1) at the midplane, for orientation
WALLS = [
    (0, 0, 0, 600), (0, 600, 0, 800), (0, 800, 1000, 800),
    (1000, 800, 1000, 0), (1000, 0, 800, 0), (800, 0, 800, 600),
    (800, 600, 200, 600), (200, 600, 200, 0), (200, 0, 0, 0),
]
DET_LABELS = [('D1', 100, 505), ('D2', 495, 700), ('D3', 900, 505),
              ('D4', 900, 295), ('D5', 900, 15)]
SRC = (100, 100)


def load_case(sp_path, res_path):
    with openmc.StatePoint(sp_path) as sp:
        t = sp.get_tally(name='mesh_flux')
        mean = t.mean.flatten().reshape(NZ, NY, NX)
        std = t.std_dev.flatten().reshape(NZ, NY, NX)
    m = mean[K_MID]
    s = std[K_MID]
    rel = np.full(m.shape, np.nan)
    ok = m > 0
    rel[ok] = s[ok] / m[ok]
    res = json.load(open(res_path))
    return m, rel, res['active_s']


def draw_geometry(ax):
    for x0, y0, x1, y1 in WALLS:
        ax.plot([x0, x1], [y0, y1], color='white', lw=0.9, alpha=0.85)
    ax.plot(*SRC, marker='*', ms=11, color='white',
            markeredgecolor='#222222', markeredgewidth=0.6, ls='none')
    for name, x, y in DET_LABELS:
        ax.annotate(name, (x, y), color='white', fontsize=8,
                    ha='center', va='center', fontweight='bold',
                    path_effects=None)


def style_axis(ax, title):
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect('equal')
    ax.set_xticks([0, 500, 1000])
    ax.set_yticks([0, 400, 800])
    ax.tick_params(length=0)


data = {name: load_case(sp, res) for name, sp, res in CASES}

# Figure 1: flux maps (log scale), shared normalization
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
all_flux = np.concatenate([data[n][0][data[n][0] > 0] for n, *_ in CASES])
norm = LogNorm(vmin=np.percentile(all_flux, 0.5), vmax=all_flux.max())
for ax, (name, *_ ) in zip(axes, CASES):
    m = np.ma.masked_less_equal(data[name][0], 0.0)
    im = ax.imshow(m, origin='lower', extent=EXTENT, norm=norm,
                   cmap='viridis', interpolation='nearest')
    im.cmap.set_bad('#e6e6e6')
    draw_geometry(ax)
    style_axis(ax, f'{name}   ({data[name][2]:.0f} s active)')
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
cb.set_label('neutron flux per source particle (track length, log scale)')
fig.suptitle('Labyrinth midplane flux (z = 130 cm). Gray = never scored. '
             'Star = Cf-252 source.', fontsize=11, color=INK)
fig.savefig(f'{OUT}/lab_flux_maps.png', dpi=150)
plt.close(fig)

# Figure 2: relative error maps, shared scale
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
for ax, (name, *_) in zip(axes, CASES):
    rel = data[name][1]
    m = np.ma.masked_invalid(rel)
    im = ax.imshow(m, origin='lower', extent=EXTENT, vmin=0.0, vmax=0.5,
                   cmap='YlOrRd', interpolation='nearest')
    im.cmap.set_bad('#b3b3b3')
    draw_geometry(ax)
    frac = np.mean(rel[np.isfinite(rel)] < 0.10)
    style_axis(
        ax, f'{name}   ({data[name][2]:.0f} s, {frac:.0%} of slice < 10%)')
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01, extend='max')
cb.set_label('relative error (1 sigma)')
fig.suptitle('Labyrinth midplane relative error. Pale = converged, dark red '
             '= noisy, gray = never scored.', fontsize=11, color=INK)
fig.savefig(f'{OUT}/lab_relerr_maps.png', dpi=150)
plt.close(fig)

# Figure 3: rel-err ratio omega/fw_cadis (matched seed and near-equal time)
fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
rf, ro = data['fw_cadis'][1], data['fw_cadis_omega'][1]
ratio = np.full(rf.shape, np.nan)
ok = np.isfinite(rf) & np.isfinite(ro) & (rf > 0)
ratio[ok] = np.log2(ro[ok] / rf[ok])
m = np.ma.masked_invalid(ratio)
im = ax.imshow(m, origin='lower', extent=EXTENT,
               norm=TwoSlopeNorm(vcenter=0.0, vmin=-2, vmax=2),
               cmap='RdBu_r', interpolation='nearest')
im.cmap.set_bad('#e6e6e6')
for x0, y0, x1, y1 in WALLS:
    ax.plot([x0, x1], [y0, y1], color='#444444', lw=0.9)
ax.plot(*SRC, marker='*', ms=11, color='#444444', ls='none')
for name, x, y in DET_LABELS:
    ax.annotate(name, (x, y), color='#222222', fontsize=8,
                ha='center', va='center', fontweight='bold')
style_axis(ax, 'Where each method wins (seed 3, equal particle count)')
cb = fig.colorbar(im, ax=ax, shrink=0.9,
                  ticks=[-2, -1, 0, 1, 2])
cb.ax.set_yticklabels(['4x lower\n(omega wins)', '2x', 'equal', '2x',
                       '4x higher\n(fw_cadis wins)'])
cb.set_label('relative error ratio, fw_cadis_omega / fw_cadis')
fig.savefig(f'{OUT}/lab_err_ratio.png', dpi=150)
plt.close(fig)

print('wrote', OUT)
