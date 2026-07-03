"""Final comparison figures across all labyrinth VR variants.

Produces:
  plots/lab_fom_comparison.png  per-detector FOM dot plot, all methods
  plots/lab_err_ratio_tier2.png win/lose maps for P3, P3-amp, P1-amp
"""
import glob
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openmc
from matplotlib.colors import TwoSlopeNorm

BASE = '/home/jon/openmc/omega_bench'
DETS = ['D1_leg1_end', 'D2_leg2_mid', 'D3_leg3_start', 'D4_leg3_mid',
        'D5_leg3_end']
DET_SHORT = ['D1\nend leg 1', 'D2\nmid leg 2', 'D3\nafter bend 2',
             'D4\nmid leg 3', 'D5\ndead end']

# Fixed categorical order and colors (Okabe-Ito, CVD safe)
METHODS = [
    ('analog', 'analog', '#000000'),
    ('fw_cadis', 'fw_cadis (scalar)', '#0072B2'),
    ('fw_cadis_omega', 'omega P1', '#E69F00'),
    ('omega_p3', 'omega P3', '#D55E00'),
    ('omega_p1_amp', 'omega P1 amp-only', '#56B4E9'),
    ('omega_p3_amp', 'omega P3 amp-only', '#CC79A7'),
    ('ang_p1', 'angular octant P1', '#009E73'),
    ('ang_p3', 'angular octant P3', '#F0E442'),
]

INK = '#333333'
plt.rcParams.update({'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': '#777777', 'ytick.color': '#777777',
                     'axes.edgecolor': '#cccccc', 'font.size': 10})

rows = []
for f in sorted(glob.glob(f'{BASE}/lab_*/result_*.json')):
    if 'pilot' in f:
        continue
    rows.append(json.load(open(f)))

# Figure 1: FOM dot plot. Per-seed dots (small) + geo-mean (large marker).
fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
ypos = np.arange(len(DETS))[::-1] * (len(METHODS) + 2)
for mi, (case, label, color) in enumerate(METHODS):
    y_off = ypos - mi
    for di, d in enumerate(DETS):
        foms = np.array([r[d]['fom'] for r in rows
                         if r['case'] == case and r[d]['fom'] > 0])
        if foms.size == 0:
            ax.annotate('no scores', (0.012, y_off[di]), fontsize=7,
                        color='#999999', va='center')
            continue
        ax.plot(foms, np.full(foms.size, y_off[di]), 'o', ms=4,
                color=color, alpha=0.45, mec='none')
        gm = np.exp(np.mean(np.log(foms)))
        ax.plot([gm], [y_off[di]], 'o', ms=9, color=color,
                mec='white', mew=1.2, label=label if di == 0 else None)
ax.set_xscale('log')
ax.set_xlim(0.01, 4000)
ax.set_yticks(ypos - (len(METHODS) - 1) / 2)
ax.set_yticklabels(DET_SHORT)
ax.set_xlabel('figure of merit at detector, 1 / (rel err$^2$ x active time)'
              '   (higher is better)')
ax.grid(axis='x', color='#e8e8e8', lw=0.7)
ax.set_axisbelow(True)
for spine in ('top', 'right', 'left'):
    ax.spines[spine].set_visible(False)
ax.legend(loc='lower right', frameon=False, fontsize=9)
ax.set_title('Labyrinth deep-detector FOM by method '
             '(small dots = seeds, large = geometric mean)', fontsize=11)
fig.savefig(f'{BASE}/plots/lab_fom_comparison.png', dpi=150)
plt.close(fig)

# Figure 2: win/lose maps for the Tier 2 variants vs fw_cadis (seed 3)
NX, NY, NZ = 60, 50, 23
EXTENT = [-100, 1100, -100, 900]
K_MID = 11
WALLS = [(0, 0, 0, 600), (0, 600, 0, 800), (0, 800, 1000, 800),
         (1000, 800, 1000, 0), (1000, 0, 800, 0), (800, 0, 800, 600),
         (800, 600, 200, 600), (200, 600, 200, 0), (200, 0, 0, 0)]
DET_XY = [('D1', 100, 505), ('D2', 495, 700), ('D3', 900, 505),
          ('D4', 900, 295), ('D5', 900, 15)]


def relerr_slice(sp_path):
    with openmc.StatePoint(sp_path) as sp:
        t = sp.get_tally(name='mesh_flux')
        m = t.mean.flatten().reshape(NZ, NY, NX)[K_MID]
        s = t.std_dev.flatten().reshape(NZ, NY, NX)[K_MID]
    rel = np.full(m.shape, np.nan)
    ok = m > 0
    rel[ok] = s[ok] / m[ok]
    return rel


ref = relerr_slice(f'{BASE}/lab_f3/statepoint.12.h5')
panels = [
    ('omega P1', f'{BASE}/lab_o3/statepoint.12.h5'),
    ('omega P3', f'{BASE}/lab_p3_3/statepoint.12.h5'),
    ('omega P1 amp-only', f'{BASE}/lab_p1amp_3/statepoint.12.h5'),
    ('omega P3 amp-only', f'{BASE}/lab_p3amp_3/statepoint.12.h5'),
    ('angular octant P1', f'{BASE}/lab_angp1_3/statepoint.12.h5'),
    ('angular octant P3', f'{BASE}/lab_angp3_3/statepoint.12.h5'),
]
fig, axes = plt.subplots(3, 2, figsize=(13, 13.8), constrained_layout=True)
for ax, (label, sp_path) in zip(axes.flat, panels):
    rel = relerr_slice(sp_path)
    ratio = np.full(ref.shape, np.nan)
    ok = np.isfinite(ref) & np.isfinite(rel) & (ref > 0)
    ratio[ok] = np.log2(rel[ok] / ref[ok])
    m = np.ma.masked_invalid(ratio)
    im = ax.imshow(m, origin='lower', extent=EXTENT,
                   norm=TwoSlopeNorm(vcenter=0.0, vmin=-2, vmax=2),
                   cmap='RdBu_r', interpolation='nearest')
    im.cmap.set_bad('#e6e6e6')
    for x0, y0, x1, y1 in WALLS:
        ax.plot([x0, x1], [y0, y1], color='#444444', lw=0.9)
    ax.plot(100, 100, marker='*', ms=10, color='#444444', ls='none')
    for name, x, y in DET_XY:
        ax.annotate(name, (x, y), color='#222222', fontsize=8, ha='center',
                    va='center', fontweight='bold')
    ax.set_title(label, fontsize=11)
    ax.set_aspect('equal')
    ax.set_xticks([0, 500, 1000])
    ax.set_yticks([0, 400, 800])
    ax.tick_params(length=0)
cb = fig.colorbar(im, ax=axes, shrink=0.75, ticks=[-2, -1, 0, 1, 2])
cb.ax.set_yticklabels(['4x lower\n(variant wins)', '2x', 'equal', '2x',
                       '4x higher\n(fw_cadis wins)'])
cb.set_label('relative error ratio, variant / fw_cadis (seed 3)')
fig.suptitle('Where each omega variant wins or loses against scalar '
             'FW-CADIS (red = worse than scalar)', fontsize=12, color=INK)
fig.savefig(f'{BASE}/plots/lab_err_ratio_tier2.png', dpi=150)
print('done')
