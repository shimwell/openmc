"""Run the CE labyrinth transport for one case and record FOM data.

Usage: run_ce_lab.py <analog|fw_cadis|fw_cadis_omega> <workdir> <particles> <batches> [ww_file] [seed]
"""
import json
import os
import sys
import time

import numpy as np
import openmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labyrinth import DETECTORS, build_model, add_tallies

case = sys.argv[1]
workdir = os.path.abspath(sys.argv[2])
particles = int(sys.argv[3])
batches = int(sys.argv[4])
ww_file = os.path.abspath(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != '-' else None
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 1

os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)
openmc.reset_auto_ids()

model, cells = build_model()
add_tallies(model, cells)
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

result = {'case': case, 'seed': seed, 'particles': particles,
          'batches': batches, 'wall_s': wall}
with openmc.StatePoint(sp_file) as sp:
    t_active = sp.runtime['active batches']
    result['active_s'] = t_active
    for name, *_ in DETECTORS:
        t = sp.get_tally(name=name)
        mean = float(t.mean.flatten()[0])
        rel = float(t.std_dev.flatten()[0] / mean) if mean > 0 else float('inf')
        fom = 1.0 / (rel ** 2 * t_active) if np.isfinite(rel) and rel > 0 else 0.0
        result[name] = {'mean': mean, 'rel_err': rel, 'fom': fom}

    mesh_t = sp.get_tally(name='mesh_flux')
    m = mesh_t.mean.flatten()
    s = mesh_t.std_dev.flatten()
    nonzero = m > 0
    rel = np.full(m.shape, np.inf)
    rel[nonzero] = s[nonzero] / m[nonzero]
    result['mesh_frac_scored'] = float(nonzero.mean())
    result['mesh_frac_rel_lt_10pct'] = float((rel < 0.10).mean())
    result['mesh_median_rel_err'] = (float(np.median(rel[nonzero]))
                                     if nonzero.any() else float('inf'))

with open(f'result_{case}_s{seed}.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2), flush=True)
