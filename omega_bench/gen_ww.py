"""Generate FW-CADIS or FW-CADIS-Omega weight windows for the bioshield.

Usage: gen_ww.py <fw_cadis|fw_cadis_omega> <workdir>

Converts the CE model to multigroup (material-wise MGXS), runs the random
ray forward+adjoint solves, and leaves ww_<method>.h5 in the workdir.
"""
import os
import shutil
import sys
import time

import openmc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bioshield import GROUP_EDGES, HALF_OUTER, build_model, make_mesh

method = sys.argv[1]
workdir = os.path.abspath(sys.argv[2])

os.makedirs(workdir, exist_ok=True)
os.chdir(workdir)
openmc.reset_auto_ids()

model, cells = build_model()

# Multigroup conversion (generates mgxs.h5 with a CE run if not present)
t0 = time.time()
model.convert_to_multigroup(
    method='material_wise', groups=GROUP_EDGES, nparticles=5000,
    correction=None)
print(f'MGXS conversion done in {time.time() - t0:.1f} s', flush=True)

# Random ray solver configuration
settings = model.settings
settings.particles = 30000
settings.batches = 150
settings.inactive = 50
settings.random_ray['ray_source'] = openmc.IndependentSource(
    space=openmc.stats.Box(
        (-HALF_OUTER, -HALF_OUTER, -HALF_OUTER),
        (HALF_OUTER, HALF_OUTER, HALF_OUTER)))
settings.random_ray['distance_inactive'] = 500.0
settings.random_ray['distance_active'] = 3000.0
settings.random_ray['volume_estimator'] = 'naive'
settings.random_ray['source_shape'] = 'flat'

# Subdivide source regions with a 20 cm mesh (matches the WW mesh)
sr_mesh = make_mesh(40)
settings.random_ray['source_region_meshes'] = [
    (sr_mesh, list(cells.values()))]

# Weight window generator on the same mesh, group-wise energy bins
ww_mesh = make_mesh(40)
settings.weight_window_generators = openmc.WeightWindowGenerator(
    method=method, mesh=ww_mesh, energy_bounds=GROUP_EDGES,
    max_realizations=settings.batches - settings.inactive)

model.tallies = openmc.Tallies([])

t0 = time.time()
sp = model.run(output=True)
print(f'{method}: RR fwd+adj done in {time.time() - t0:.1f} s -> {sp}',
      flush=True)

shutil.copy('weight_windows.h5', f'ww_{method}.h5')
print(f'saved ww_{method}.h5', flush=True)
