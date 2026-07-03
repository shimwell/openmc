"""Compare the fw_cadis and fw_cadis_omega weight window maps.

Reports the omega/fw_cadis lower-bound shape ratio as a function of depth
into the shield (distance from cavity wall along the x axis at the source
midplane) per energy group, plus global ratio statistics.
"""
import numpy as np
import openmc

BASE = '/home/jon/openmc/omega_bench'

ww_f = openmc.WeightWindowsList.from_hdf5(f'{BASE}/ww_f/ww_fw_cadis.h5')[0]
ww_o = openmc.WeightWindowsList.from_hdf5(
    f'{BASE}/ww_o/ww_fw_cadis_omega.h5')[0]

# lower_ww_bounds shape: (n_energy, nx*ny*nz) -> reshape to (E, 40, 40, 40)
lb_f = np.asarray(ww_f.lower_ww_bounds)
lb_o = np.asarray(ww_o.lower_ww_bounds)
print('raw shapes:', lb_f.shape, lb_o.shape)
n_e = lb_f.shape[-1] if lb_f.shape[0] == 40 else lb_f.shape[0]

# openmc stores (mesh_i, mesh_j, mesh_k, energy) for WeightWindows python API
if lb_f.shape[0] == 40:
    lb_f = np.moveaxis(lb_f, -1, 0)
    lb_o = np.moveaxis(lb_o, -1, 0)
lb_f = lb_f.reshape(n_e, 40, 40, 40)
lb_o = lb_o.reshape(n_e, 40, 40, 40)

valid = (lb_f > 0) & (lb_o > 0)
print(f'valid bins: fw_cadis={np.sum(lb_f > 0)}, omega={np.sum(lb_o > 0)}, '
      f'both={valid.sum()} of {lb_f.size}')

ratio = np.where(valid, lb_o / np.where(lb_f > 0, lb_f, 1.0), np.nan)
print(f'global ratio stats (unnormalized): '
      f'median={np.nanmedian(ratio):.3f} '
      f'p5={np.nanpercentile(ratio, 5):.3f} '
      f'p95={np.nanpercentile(ratio, 95):.3f}')

# Depth profile along +x at the midplane (j=k=20), from cavity wall (x=200,
# bin 30) to outer edge (x=400, bin 39). Mesh bins are 20 cm.
print('\nDepth profile at midplane (per group, ratio normalized at first '
      'shield bin):')
xbins = np.arange(30, 40)
depths = (xbins - 30) * 20 + 10
hdr = 'group ' + ' '.join(f'{d:7.0f}' for d in depths)
print(hdr + '   (depth into shield, cm)')
for e in range(n_e):
    prof_f = lb_f[e, xbins, 20, 20]
    prof_o = lb_o[e, xbins, 20, 20]
    ok = (prof_f > 0) & (prof_o > 0)
    if not ok[0]:
        print(f'{e:5d}  (first bin invalid, skipped)')
        continue
    r = (prof_o / prof_o[0]) / (prof_f / prof_f[0])
    row = ' '.join(f'{v:7.3f}' if k else '      -' for v, k in zip(r, ok))
    print(f'{e:5d}  {row}')

# Decades of WW range covered in the deepest group with valid data
for e in range(n_e):
    v = lb_f[e][lb_f[e] > 0]
    w = lb_o[e][lb_o[e] > 0]
    if v.size and w.size:
        print(f'group {e}: fw_cadis range {np.log10(v.max()/v.min()):5.2f} '
              f'decades, omega range {np.log10(w.max()/w.min()):5.2f} decades')
