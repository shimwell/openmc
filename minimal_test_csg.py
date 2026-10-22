"""Runs a small simulation to check that a wheel can run OpenMC and not just
import it.

minimal_test.py covers DAGMC, unstructured meshes and moab, none of which are
in the vanilla Windows wheel, so this test sticks to CSG geometry and the two
nuclear data files committed next to it. It deliberately goes through
model.run() rather than openmc.lib so that the openmc executable and the
console script that launches it are both exercised.

The data files are found relative to this file so that the test can be run from
a working directory that does not contain the openmc source, which would
otherwise shadow the installed package.
"""

from pathlib import Path

import openmc

this_dir = Path(__file__).resolve().parent
openmc.config['cross_sections'] = this_dir / 'cross_sections.xml'

breeder_mat = openmc.Material()
breeder_mat.add_nuclide('Li7', 1.0)
breeder_mat.set_density('g/cm3', 0.5)

sphere_surf = openmc.Sphere(r=100, boundary_type='vacuum')
sphere_cell = openmc.Cell(region=-sphere_surf, fill=breeder_mat)

my_source = openmc.IndependentSource()
my_source.space = openmc.stats.Point((0, 0, 0))
my_source.energy = openmc.stats.Discrete([14e6], [1.0])

settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.batches = 3
settings.particles = 1000
settings.source = my_source
# the photon data for Li is committed alongside the neutron data so this also
# checks that photon transport works
settings.photon_transport = True

my_tally = openmc.Tally(name='tritium_production')
my_tally.scores = ['H3-production']

my_model = openmc.Model(
    materials=openmc.Materials([breeder_mat]),
    geometry=openmc.Geometry([sphere_cell]),
    settings=settings,
    tallies=openmc.Tallies([my_tally]),
)

statepoint_file = my_model.run()

with openmc.StatePoint(statepoint_file) as statepoint:
    tritium_production = statepoint.get_tally(name='tritium_production')
    result = tritium_production.mean.flatten()[0]

print(f'tritium production per source neutron {result}')

assert result > 0.0

print('minimal CSG simulation completed')
