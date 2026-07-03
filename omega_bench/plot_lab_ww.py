"""Midplane map of how fw_cadis_omega reshaped the weight windows."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openmc
from matplotlib.colors import TwoSlopeNorm

BASE = '/home/jon/openmc/omega_bench'
NX, NY, NZ = 60, 50, 23
EXTENT = [-100, 1100, -100, 900]
K_MID = 11
G = 8  # 6-10 MeV group

INK = '#333333'
plt.rcParams.update({'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': '#777777', 'ytick.color': '#777777',
                     'axes.edgecolor': '#cccccc', 'font.size': 10})

lb_f = np.asarray(openmc.WeightWindowsList.from_hdf5(
    f'{BASE}/lab_ww_f/ww_fw_cadis.h5')[0].lower_ww_bounds)
lb_o = np.asarray(openmc.WeightWindowsList.from_hdf5(
    f'{BASE}/lab_ww_o/ww_fw_cadis_omega.h5')[0].lower_ww_bounds)
# stored as (nx, ny, nz, n_energy)
f = lb_f[:, :, K_MID, G].T  # -> [y, x]
o = lb_o[:, :, K_MID, G].T

ratio = np.full(f.shape, np.nan)
ok = (f > 0) & (o > 0)
ratio[ok] = np.log2(o[ok] / f[ok])

WALLS = [(0, 0, 0, 600), (0, 600, 0, 800), (0, 800, 1000, 800),
         (1000, 800, 1000, 0), (1000, 0, 800, 0), (800, 0, 800, 600),
         (800, 600, 200, 600), (200, 600, 200, 0), (200, 0, 0, 0)]
DETS = [('D1', 100, 505), ('D2', 495, 700), ('D3', 900, 505),
        ('D4', 900, 295), ('D5', 900, 15)]

fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
m = np.ma.masked_invalid(ratio)
im = ax.imshow(m, origin='lower', extent=EXTENT,
               norm=TwoSlopeNorm(vcenter=0.0, vmin=-2, vmax=2),
               cmap='RdBu_r', interpolation='nearest')
im.cmap.set_bad('#e6e6e6')
for x0, y0, x1, y1 in WALLS:
    ax.plot([x0, x1], [y0, y1], color='#444444', lw=0.9)
ax.plot(100, 100, marker='*', ms=11, color='#444444', ls='none')
for name, x, y in DETS:
    ax.annotate(name, (x, y), color='#222222', fontsize=8, ha='center',
                va='center', fontweight='bold')
ax.set_title('How omega reshaped the weight windows (6-10 MeV group)',
             fontsize=11)
ax.set_aspect('equal')
ax.set_xticks([0, 500, 1000])
ax.set_yticks([0, 400, 800])
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.9, ticks=[-2, -1, 0, 1, 2])
cb.ax.set_yticklabels(['4x lower bound\n(omega splits more)', '2x', 'same',
                       '2x', '4x higher bound\n(omega roulettes more)'])
cb.set_label('WW lower bound ratio, fw_cadis_omega / fw_cadis')
fig.savefig(f'{BASE}/plots/lab_ww_ratio.png', dpi=150)
print('done')
