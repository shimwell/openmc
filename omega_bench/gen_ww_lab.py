"""Generate FW-CADIS or FW-CADIS-Omega weight windows for the labyrinth.

Usage: gen_ww_lab.py <fw_cadis|fw_cadis_omega> <workdir> [order] [clamp_min]
"""
import os
import shutil
import sys
import time

import numpy as np
import openmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labyrinth import (GROUP_EDGES, SRC_XYZ, X_MIN, X_MAX, Y_MIN, Y_MAX,
                       Z_MIN, Z_MAX, build_model, make_mesh)

method = sys.argv[1]
workdir = os.path.abspath(sys.argv[2])
order = int(sys.argv[3]) if len(sys.argv) > 3 else None
clamp_min = (float(sys.argv[4])
             if len(sys.argv) > 4 and sys.argv[4] != '-' else None)
angular = int(sys.argv[5]) if len(sys.argv) > 5 else None
os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)
openmc.reset_auto_ids()

model, cells = build_model()

t0 = time.time()
model.convert_to_multigroup(
    method='material_wise', groups=GROUP_EDGES, nparticles=5000,
    correction=None)
print(f'MGXS conversion done in {time.time() - t0:.1f} s', flush=True)

# Random ray requires a discrete (multigroup) source energy distribution:
# collapse the Cf-252 Watt spectrum onto the group structure
rng = np.random.default_rng(42)
a, b = 1.18e6, 1.03419e-6
# Rejection-free Watt sampling: E = w + a^2 b/4 + z*sqrt(a^2 b w), w ~ Exp(a)
w = rng.exponential(a, size=1_000_000)
z = rng.standard_normal(1_000_000)
E = w + a * a * b / 4.0 + z * np.sqrt(a * a * b * w)
E = E[(E > 0) & (E < GROUP_EDGES[-1])]
hist, _ = np.histogram(E, bins=GROUP_EDGES)
probs = hist / hist.sum()
mids = [0.5 * (GROUP_EDGES[i] + GROUP_EDGES[i + 1])
        for i in range(len(GROUP_EDGES) - 1)]
keep = probs > 0
mg_source = openmc.IndependentSource(
    space=openmc.stats.Point(SRC_XYZ),
    angle=openmc.stats.Isotropic(),
    energy=openmc.stats.Discrete(
        [m for m, k in zip(mids, keep) if k],
        list(probs[keep])))
model.settings.source = mg_source
print('MG source group probabilities:',
      np.array2string(probs, precision=4), flush=True)

settings = model.settings
settings.particles = 20000
settings.batches = 150
settings.inactive = 50
settings.random_ray['ray_source'] = openmc.IndependentSource(
    space=openmc.stats.Box((X_MIN, Y_MIN, Z_MIN), (X_MAX, Y_MAX, Z_MAX)))
settings.random_ray['distance_inactive'] = 500.0
settings.random_ray['distance_active'] = 2500.0
settings.random_ray['volume_estimator'] = 'naive'
settings.random_ray['source_shape'] = 'flat'
settings.random_ray['source_region_meshes'] = [
    (make_mesh(), list(cells.values()))]

wwg = openmc.WeightWindowGenerator(
    method=method, mesh=make_mesh(), energy_bounds=GROUP_EDGES,
    max_realizations=settings.batches - settings.inactive)
params = {}
if order is not None:
    params['order'] = order
if clamp_min is not None:
    params['clamp_min'] = clamp_min
if angular is not None:
    params['angular_bins'] = angular
if params:
    wwg.update_parameters = params
settings.weight_window_generators = wwg

model.tallies = openmc.Tallies([])

t0 = time.time()
sp = model.run(output=True)
print(f'{method}: RR fwd+adj done in {time.time() - t0:.1f} s -> {sp}',
      flush=True)
shutil.copy('weight_windows.h5', f'ww_{method}.h5')
print(f'saved ww_{method}.h5', flush=True)
