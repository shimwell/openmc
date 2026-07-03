"""Run the CE bioshield transport for one case and record FOM data.

Usage: run_ce.py <analog|fw_cadis|fw_cadis_omega> <workdir> <particles> <batches> [ww_file] [seed]
"""
import json
import os
import sys
import time

import numpy as np
import openmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bioshield import build_model, add_tallies

case = sys.argv[1]
workdir = os.path.abspath(sys.argv[2])
particles = int(sys.argv[3])
batches = int(sys.argv[4])
ww_file = os.path.abspath(sys.argv[5]) if len(sys.argv) > 5 else None
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1

os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)
openmc.reset_auto_ids()

model, cells = build_model()
add_tallies(model, cells, mesh_n=40)
model.settings.particles = particles
model.settings.batches = batches
model.settings.seed = seed

if case != 'analog':
    wws = openmc.WeightWindowsList.from_hdf5(ww_file)
    model.settings.weight_windows = list(wws)
    model.settings.weight_windows_on = True

t0 = time.time()
sp_file = model.run(output=True)
wall = time.time() - t0

with openmc.StatePoint(sp_file) as sp:
    t_active = sp.runtime['active batches']
    det = sp.get_tally(name='det_flux')
    det_mean = float(det.mean.flatten()[0])
    det_rel = float(det.std_dev.flatten()[0] / det_mean) if det_mean > 0 else float('inf')

    mesh_t = sp.get_tally(name='mesh_flux')
    m = mesh_t.mean.flatten()
    s = mesh_t.std_dev.flatten()
    nonzero = m > 0
    rel = np.full(m.shape, np.inf)
    rel[nonzero] = s[nonzero] / m[nonzero]
    frac_scored = float(nonzero.mean())
    frac_conv = float((rel < 0.10).mean())
    median_rel = float(np.median(rel[nonzero])) if nonzero.any() else float('inf')

fom_det = 1.0 / (det_rel ** 2 * t_active) if np.isfinite(det_rel) and det_rel > 0 else 0.0

result = {
    'case': case,
    'seed': seed,
    'particles': particles,
    'batches': batches,
    'wall_s': wall,
    'active_s': t_active,
    'det_mean': det_mean,
    'det_rel_err': det_rel,
    'det_fom': fom_det,
    'mesh_frac_scored': frac_scored,
    'mesh_frac_rel_lt_10pct': frac_conv,
    'mesh_median_rel_err': median_rel,
}
with open(f'result_{case}_s{seed}.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2), flush=True)
