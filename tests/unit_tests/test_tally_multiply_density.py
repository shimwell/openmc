import numpy as np
import openmc
import pytest


def test_micro_macro_compare(run_in_tmpdir):
    # Create simple sphere model with H1 and H2
    mat = openmc.Material()
    mat.add_components({'H1': 1.0, 'H2': 1.0})
    mat.set_density('g/cm3', 1.0)
    sph = openmc.Sphere(r=10.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-sph)
    model = openmc.Model()
    model.geometry = openmc.Geometry([cell])
    model.settings.run_mode = 'fixed source'
    model.settings.particles = 1000
    model.settings.batches = 10

    # Set up two reaction rate tallies, one that multplies by density and the
    # other that doesn't
    tally_macro = openmc.Tally()
    tally_macro.nuclides = ['H1', 'H2', 'H3']
    tally_macro.scores = ['total', 'elastic']
    tally_micro = openmc.Tally()
    tally_micro.nuclides = ['H1', 'H2', 'H3']
    tally_micro.scores = ['total', 'elastic']
    tally_micro.multiply_density = False
    model.tallies = [tally_macro, tally_micro]

    sp_filename = model.run()
    with openmc.StatePoint(sp_filename) as sp:
        tally_macro = sp.tallies[tally_macro.id]
        tally_micro = sp.tallies[tally_micro.id]

    # Make sure multply_density attribute from statepoint is set correctly
    assert tally_macro.multiply_density
    assert not tally_micro.multiply_density

    # Dividing macro by density should give micro
    density = mat.get_nuclide_atom_densities()
    for nuc in ('H1', 'H2'):
        micro_derived = tally_macro.get_values(nuclides=[nuc]) / density[nuc]
        micro = tally_micro.get_values(nuclides=[nuc])
        assert micro_derived == pytest.approx(micro)

    # For macro tally, H3 scores should be zero
    assert np.all(tally_macro.get_values(nuclides=['H3']) == 0.0)

    # For micro tally, H3 scores should be positive
    assert np.all(tally_micro.get_values(nuclides=['H3']) > 0.0)


def test_fission_in_void(run_in_tmpdir):
    """Test that fission scores work in void with multiply_density=False.

    The macro_xs().fission guard must not skip microscopic fission scoring
    when atom_density has been set to 1.0 by the void path.
    """
    sph = openmc.Sphere(r=10.0, boundary_type='vacuum')
    void_cell = openmc.Cell(region=-sph)
    model = openmc.Model()
    model.geometry = openmc.Geometry([void_cell])
    model.settings.run_mode = 'fixed source'
    model.settings.particles = 1000
    model.settings.batches = 5
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point()
    )

    tally = openmc.Tally()
    tally.scores = ['fission', 'nu-fission']
    tally.nuclides = ['U235']
    tally.multiply_density = False
    model.tallies = [tally]

    sp_filename = model.run()
    with openmc.StatePoint(sp_filename) as sp:
        t = sp.tallies[tally.id]

    assert np.all(t.mean > 0.0)
