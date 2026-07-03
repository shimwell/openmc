"""Shared model builder for the 2 m concrete bioshield benchmark.

Geometry (cm): a cubic cavity of half-width 200 surrounded by a 2 m thick
ordinary-concrete shield (outer half-width 400, vacuum boundary). A DT ring
source (14.06 MeV, isotropic) of radius 150 sits at the cavity midplane,
represented by a thin annular void cell so that the random ray solver can
constrain the volumetric source to it. The outermost 10 cm of concrete
(390 to 400) is a separate detector shell cell used for the deep tally.
"""
import numpy as np
import openmc

# Energy group boundaries (eV) used for the MGXS library and the weight
# window energy bins. Top edge covers the 14.06 MeV source.
GROUP_EDGES = [0.0, 0.625, 1.0e2, 1.0e4, 1.0e5, 5.0e5, 1.0e6, 3.0e6,
               6.0e6, 1.0e7, 1.5e7]

HALF_CAVITY = 200.0
HALF_OUTER = 400.0
DET_INNER = 390.0
RING_R = 150.0
SOURCE_E = 14.06e6


def make_mesh(n):
    mesh = openmc.RegularMesh()
    mesh.dimension = (n, n, n)
    mesh.lower_left = (-HALF_OUTER, -HALF_OUTER, -HALF_OUTER)
    mesh.upper_right = (HALF_OUTER, HALF_OUTER, HALF_OUTER)
    return mesh


def build_model():
    concrete = openmc.Material(name='concrete')
    concrete.set_density('g/cm3', 2.3)
    # NIST ordinary concrete, weight fractions
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

    materials = openmc.Materials([concrete])

    # Surfaces
    inner_box = openmc.model.RectangularParallelepiped(
        -HALF_CAVITY, HALF_CAVITY, -HALF_CAVITY, HALF_CAVITY,
        -HALF_CAVITY, HALF_CAVITY)
    det_box = openmc.model.RectangularParallelepiped(
        -DET_INNER, DET_INNER, -DET_INNER, DET_INNER, -DET_INNER, DET_INNER)
    outer_box = openmc.model.RectangularParallelepiped(
        -HALF_OUTER, HALF_OUTER, -HALF_OUTER, HALF_OUTER,
        -HALF_OUTER, HALF_OUTER, boundary_type='vacuum')

    ring_in = openmc.ZCylinder(r=RING_R - 10.0)
    ring_out = openmc.ZCylinder(r=RING_R + 10.0)
    ring_zlo = openmc.ZPlane(-10.0)
    ring_zhi = openmc.ZPlane(10.0)
    ring_region = +ring_in & -ring_out & +ring_zlo & -ring_zhi

    ring_cell = openmc.Cell(name='ring', region=ring_region)
    cavity_cell = openmc.Cell(name='cavity', region=-inner_box & ~ring_region)
    shield_cell = openmc.Cell(
        name='shield', fill=concrete, region=+inner_box & -det_box)
    det_cell = openmc.Cell(
        name='detector', fill=concrete, region=+det_box & -outer_box)

    geometry = openmc.Geometry(
        [ring_cell, cavity_cell, shield_cell, det_cell])

    settings = openmc.Settings()
    settings.run_mode = 'fixed source'

    source = openmc.IndependentSource(
        space=openmc.stats.CylindricalIndependent(
            r=openmc.stats.Discrete([RING_R], [1.0]),
            phi=openmc.stats.Uniform(0.0, 2.0 * np.pi),
            z=openmc.stats.Discrete([0.0], [1.0])),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Discrete([SOURCE_E], [1.0]),
        constraints={'domains': [ring_cell]})
    settings.source = source

    model = openmc.Model(geometry=geometry, materials=materials,
                         settings=settings)

    cells = {c.name: c for c in [ring_cell, cavity_cell, shield_cell,
                                 det_cell]}
    return model, cells


def add_tallies(model, cells, mesh_n=40):
    det_tally = openmc.Tally(name='det_flux')
    det_tally.filters = [openmc.CellFilter([cells['detector']])]
    det_tally.scores = ['flux']

    mesh_tally = openmc.Tally(name='mesh_flux')
    mesh_tally.filters = [openmc.MeshFilter(make_mesh(mesh_n))]
    mesh_tally.scores = ['flux']

    model.tallies = openmc.Tallies([det_tally, mesh_tally])
