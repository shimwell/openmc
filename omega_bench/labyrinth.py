"""Protvino-like 3-section concrete labyrinth (ALARM-CF-AIR-LAB-001 style).

U-shaped corridor (two 90 degree bends) through a concrete monolith:

  y=9 +-----------------------------------------+
      |  ################ concrete ############ |
  y=8 |  #  +----------- leg 2 -----------+  #  |
      |  #  | J1                       J2 |  #  |
  y=6 |  #  +--+                       +--+  #  |
      |  #  |l |                       | l |  #  |
      |  #  |e |     concrete          | e |  #  |
      |  #  |g |      island           | g |  #  |
      |  #  |1 |                       | 3 |  #  |
      |  #  |* |                       |   |  #  | * = Cf-252 source
  y=0 |  #  +--+-----------------------+---+  #  |
  y=-1+-----------------------------------------+
      x=-1 x=0 x=2                    x=8 x=10 x=11

Corridor cross-section 2.0 m wide x 2.6 m high (z in [0, 260] cm), 100 cm
of concrete on all sides (x in [-100, 1100], y in [-100, 900],
z in [-100, 360]), vacuum outside. Leg 1 runs +y (x in [0,200]), leg 2
runs +x (y in [600,800]), leg 3 runs -y (x in [800,1000]) and dead-ends
at y=0. A Cf-252 point source (Watt spectrum) sits on the leg 1 axis
100 cm from the entrance dead end. Five 10 cm thick volumetric void
detectors span the corridor: D1 end of leg 1, D2 middle of leg 2, D3
start of leg 3 (after the second bend), D4 middle of leg 3, D5 dead end
of leg 3.
"""
import numpy as np
import openmc

GROUP_EDGES = [0.0, 0.625, 1.0e2, 1.0e4, 1.0e5, 5.0e5, 1.0e6, 3.0e6,
               6.0e6, 1.0e7, 1.5e7]

X_MIN, X_MAX = -100.0, 1100.0
Y_MIN, Y_MAX = -100.0, 900.0
Z_MIN, Z_MAX = -100.0, 360.0
Z_LO, Z_HI = 0.0, 260.0
SRC_XYZ = (100.0, 100.0, 130.0)

# (name, xlo, xhi, ylo, yhi) of the 10 cm detector slabs, corridor z extent
DETECTORS = [
    ('D1_leg1_end', 0.0, 200.0, 500.0, 510.0),
    ('D2_leg2_mid', 490.0, 500.0, 600.0, 800.0),
    ('D3_leg3_start', 800.0, 1000.0, 500.0, 510.0),
    ('D4_leg3_mid', 800.0, 1000.0, 290.0, 300.0),
    ('D5_leg3_end', 800.0, 1000.0, 10.0, 20.0),
]


def concrete_material():
    concrete = openmc.Material(name='concrete')
    concrete.set_density('g/cm3', 2.3)
    concrete.add_element('H', 0.022100, 'wo')
    concrete.add_element('C', 0.002484, 'wo')
    concrete.add_element('O', 0.574930, 'wo')
    concrete.add_element('Na', 0.015208, 'wo')
    concrete.add_element('Mg', 0.001266, 'wo')
    concrete.add_element('Al', 0.019953, 'wo')
    concrete.add_element('Si', 0.304627, 'wo')
    concrete.add_element('K', 0.010045, 'wo')
    concrete.add_element('Ca', 0.042951, 'wo')
    concrete.add_element('Fe', 0.006435, 'wo')
    return concrete


def make_mesh():
    mesh = openmc.RegularMesh()
    mesh.dimension = (60, 50, 23)  # 20 cm voxels
    mesh.lower_left = (X_MIN, Y_MIN, Z_MIN)
    mesh.upper_right = (X_MAX, Y_MAX, Z_MAX)
    return mesh


def _box(xlo, xhi, ylo, yhi, zlo=Z_LO, zhi=Z_HI):
    return openmc.model.RectangularParallelepiped(xlo, xhi, ylo, yhi, zlo, zhi)


def build_model():
    concrete = concrete_material()
    materials = openmc.Materials([concrete])

    outer = openmc.model.RectangularParallelepiped(
        X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, boundary_type='vacuum')

    # Corridor sections (void)
    leg1 = _box(0.0, 200.0, 0.0, 600.0)
    leg2 = _box(0.0, 1000.0, 600.0, 800.0)  # includes both junctions
    leg3 = _box(800.0, 1000.0, 0.0, 600.0)
    corridor_region = -leg1 | -leg2 | -leg3

    det_cells = []
    det_regions = None
    for name, xlo, xhi, ylo, yhi in DETECTORS:
        r = -_box(xlo, xhi, ylo, yhi)
        det_cells.append(openmc.Cell(name=name, region=r))
        det_regions = r if det_regions is None else det_regions | r

    corridor_cell = openmc.Cell(
        name='corridor', region=corridor_region & ~det_regions)
    concrete_cell = openmc.Cell(
        name='walls', fill=concrete, region=-outer & ~corridor_region)

    geometry = openmc.Geometry([corridor_cell, concrete_cell] + det_cells)

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'

    # Cf-252 spontaneous fission Watt spectrum
    source = openmc.IndependentSource(
        space=openmc.stats.Point(SRC_XYZ),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Watt(a=1.18e6, b=1.03419e-6))
    settings.source = source

    model = openmc.Model(geometry=geometry, materials=materials,
                         settings=settings)

    cells = {c.name: c for c in [corridor_cell, concrete_cell] + det_cells}
    return model, cells


def add_tallies(model, cells):
    tallies = []
    for name, *_ in DETECTORS:
        t = openmc.Tally(name=name)
        t.filters = [openmc.CellFilter([cells[name]])]
        t.scores = ['flux']
        tallies.append(t)

    mesh_tally = openmc.Tally(name='mesh_flux')
    mesh_tally.filters = [openmc.MeshFilter(make_mesh())]
    mesh_tally.scores = ['flux']
    tallies.append(mesh_tally)

    model.tallies = openmc.Tallies(tallies)
