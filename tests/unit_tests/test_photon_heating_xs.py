"""Photon heating scored from the photon heating cross section.

Photon heating in nuclide bins with multiply_density off is scored as the flux
times a heating cross section that OpenMC computes for each element, so that it
gives a per-atom result that can be scored where the element is absent.
"""

import openmc
import pytest


def _photon_source(energy):
    return openmc.IndependentSource(
        space=openmc.stats.Point(),
        energy=openmc.stats.Discrete([energy], [1.0]),
        particle='photon')


def test_heating_of_an_absent_element(run_in_tmpdir):
    """Heating per atom is the same whether or not the element is present.

    Uncollided photons all have the source energy, so their heating divided by
    their flux is the heating cross section at that energy. It must be the
    same in the inner iron sphere, which holds no silicon, as in the silicon
    shell around it.
    """
    openmc.reset_auto_ids()

    iron = openmc.Material()
    iron.add_nuclide('Fe56', 1.0)
    iron.set_density('g/cm3', 0.1)

    silicon = openmc.Material()
    silicon.add_nuclide('Si28', 1.0)
    silicon.set_density('g/cm3', 0.1)

    inner = openmc.Sphere(r=5.0)
    outer = openmc.Sphere(r=10.0, boundary_type='vacuum')
    iron_cell = openmc.Cell(fill=iron, region=-inner)
    silicon_cell = openmc.Cell(fill=silicon, region=+inner & -outer)

    model = openmc.Model()
    model.geometry = openmc.Geometry([iron_cell, silicon_cell])
    model.settings.run_mode = 'fixed source'
    model.settings.photon_transport = True
    model.settings.particles = 200
    model.settings.batches = 2
    model.settings.source = _photon_source(1.0e6)

    filters = [
        openmc.CellFilter([iron_cell, silicon_cell]),
        openmc.EnergyFilter([0.999e6, 1.001e6]),
    ]
    flux = openmc.Tally()
    flux.filters = filters
    flux.scores = ['flux']

    heating = openmc.Tally()
    heating.filters = filters
    heating.nuclides = ['Si28']
    heating.scores = ['heating']
    heating.multiply_density = False
    model.tallies = openmc.Tallies([flux, heating])

    sp_path = model.run(apply_tally_results=True)

    # Heating from the cross section keeps the tracklength estimator
    with openmc.StatePoint(sp_path) as sp:
        assert sp.tallies[heating.id].estimator == 'tracklength'

    iron_flux, silicon_flux = flux.mean.ravel()
    iron_heating, silicon_heating = heating.mean.ravel()
    assert silicon_heating > 0.0
    assert iron_heating / iron_flux == pytest.approx(
        silicon_heating / silicon_flux, rel=1e-10)


def test_heating_matches_energy_balance(run_in_tmpdir):
    """Applying the material itself recovers the energy balance heating."""
    openmc.reset_auto_ids()

    silicon = openmc.Material()
    silicon.add_element('Si', 1.0)
    silicon.set_density('g/cm3', 2.329)

    sphere = openmc.Sphere(r=20.0, boundary_type='vacuum')
    model = openmc.Model()
    model.geometry = openmc.Geometry(
        [openmc.Cell(fill=silicon, region=-sphere)])
    model.settings.run_mode = 'fixed source'
    model.settings.photon_transport = True
    model.settings.particles = 10000
    model.settings.batches = 5
    model.settings.source = _photon_source(1.0e6)

    photon = openmc.ParticleFilter('photon')
    energy_balance = openmc.Tally()
    energy_balance.filters = [photon]
    energy_balance.scores = ['heating']

    from_xs = openmc.Tally()
    from_xs.filters = [photon]
    from_xs.nuclides = silicon.get_nuclides()
    from_xs.scores = ['heating']
    from_xs.multiply_density = False
    model.tallies = openmc.Tallies([energy_balance, from_xs])

    model.run(apply_tally_results=True)
    from_xs.apply_virtual_material(silicon)

    assert from_xs.mean.sum() == pytest.approx(
        energy_balance.mean.sum(), rel=0.02)


def test_analog_estimator_rejected(run_in_tmpdir):
    openmc.reset_auto_ids()

    silicon = openmc.Material()
    silicon.add_nuclide('Si28', 1.0)
    silicon.set_density('g/cm3', 2.329)

    sphere = openmc.Sphere(r=5.0, boundary_type='vacuum')
    model = openmc.Model()
    model.geometry = openmc.Geometry(
        [openmc.Cell(fill=silicon, region=-sphere)])
    model.settings.run_mode = 'fixed source'
    model.settings.photon_transport = True
    model.settings.particles = 100
    model.settings.batches = 1
    model.settings.source = _photon_source(1.0e6)

    tally = openmc.Tally()
    tally.nuclides = ['Si28']
    tally.scores = ['heating']
    tally.multiply_density = False
    tally.estimator = 'analog'
    model.tallies = openmc.Tallies([tally])

    with pytest.raises(RuntimeError, match='tracklength or collision'):
        model.run()
