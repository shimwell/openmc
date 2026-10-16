"""Check example schedules and RegularMesh plotting without transport."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import openmc
from openmc.deplete import d1s


EXAMPLES = Path(__file__).resolve().parents[2] / 'examples' / 'd1s_moving'


@pytest.fixture
def examples(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return SimpleNamespace(
        common=importlib.import_module('d1s_mesh_common'),
        moving_source=importlib.import_module('moving_neutron_source'),
        moving_volume=importlib.import_module('moving_activated_volume'))


@pytest.fixture
def args():
    return SimpleNamespace(batches=3, particles=100, mesh_bins=(3, 2),
                           chain_file=None)


def test_mesh_axis_order_and_volume_normalization(examples):
    mesh = openmc.RegularMesh()
    mesh.dimension = (3, 2, 1)
    mesh.lower_left = (0., 0., 0.)
    mesh.upper_right = (6., 6., 4.)
    tally = openmc.Tally()
    tally.filters = [openmc.MeshFilter(mesh), openmc.ParticleFilter('photon')]
    tally.nuclides = ['total']
    tally.scores = ['flux']
    tally._derived = True
    # OpenMC's flat mesh ordering is x fastest, then y, then z.
    physical = np.array([1., 2., 3., 11., 12., 13.])
    tally._mean = (physical * 24.).reshape(6, 1, 1)
    tally._std_dev = tally._mean * 0.1
    expected = np.array([[1., 11.], [2., 12.], [3., 13.]])
    np.testing.assert_array_equal(examples.common.mesh_flux(tally), expected)
    np.testing.assert_allclose(
        examples.common.mesh_flux(tally, 'std_dev'), expected * 0.1)


@pytest.mark.parametrize('scenario', ['moving_source', 'moving_volume'])
def test_example_schedules(examples, args, scenario):
    module = getattr(examples, scenario)
    simulation, positions = module.make_simulation(args)
    simulation.validate()
    irradiations = [step for step in simulation.steps
                   if isinstance(step, d1s.Irradiation)]
    cooling = [step for step in simulation.steps
               if isinstance(step, d1s.Cooling)]
    assert len(irradiations) == 3
    assert sum(step.duration for step in irradiations) == 3600.
    assert sum(len(step.sample_times) for step in cooling) == 12
    source_positions = [
        tuple(step.configuration.model.settings.source[0].space.xyz)
        for step in irradiations]
    emitter_positions = [step.configuration.transforms['emitter'][0, 3]
                         for step in irradiations]
    if scenario == 'moving_source':
        assert source_positions == list(module.NEUTRON_POSITIONS_CM)
        assert emitter_positions == [0., 0., 0.]
        assert positions == {'stationary-emitter': 0.}
    else:
        assert source_positions == [module.NEUTRON_POSITION_CM] * 3
        assert tuple(emitter_positions) == module.IRRADIATION_POSITIONS_CM
        assert tuple(positions.values()) == module.COOLING_POSITIONS_CM
    for step in simulation.steps:
        configuration = step.configuration
        assert list(configuration.components) == ['emitter']
        tally = configuration.model.tallies[0]
        mesh = tally.find_filter(openmc.MeshFilter).mesh
        assert isinstance(mesh, openmc.RegularMesh)
        assert tuple(mesh.dimension) == (3, 2, 1)
        assert tuple(mesh.lower_left) == examples.common.MESH_LOWER_LEFT_CM
        assert tuple(mesh.upper_right) == examples.common.MESH_UPPER_RIGHT_CM
        cell = configuration.components['emitter'].cells[0]
        assert (cell.region.surface.x0
                == configuration.transforms['emitter'][0, 3])
    assert all(len(step.configuration.model.settings.source) == 0
               for step in cooling)