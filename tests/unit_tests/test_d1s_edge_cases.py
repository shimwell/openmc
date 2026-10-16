"""Transport reproducers for D1S normalization, placement, and ancestry edges.

Photon replay cases may prescribe a synthetic production bank. Their
weighted FileSource sampling, photon transport, and tally scoring are real;
all other banks are captured from neutron transport. No locator is mocked.
"""

from copy import deepcopy
from pathlib import Path
import shutil

import h5py
import lxml.etree as ET
import numpy as np
import pytest

import openmc
import openmc.lib
from openmc.deplete import d1s, _d1s
from tests.unit_tests.test_d1s_multimodel import (
    CAD, CHAIN, PARENTS, _factor, _simulation, _sites, _snapshot, _tally,
    _vectors, _write_capture,
)


@pytest.fixture(scope='module')
def transport_options():
    executable = shutil.which('openmc')
    if executable is None:
        pytest.skip('OpenMC executable is unavailable')
    xs = openmc.config.get('cross_sections')
    if xs is None or not Path(xs).is_file():
        pytest.skip('Continuous-energy cross sections are unavailable')
    library = openmc.data.DataLibrary.from_xml(xs)
    for name, kind in (('Ni64', 'neutron'), ('Ni', 'photon')):
        entry = library.get_by_material(name, kind)
        if entry is None or not Path(entry['path']).is_file():
            pytest.skip(f'{kind} data for {name} are unavailable')
    return {'openmc_exec': executable, 'threads': 2, 'output': False}


@pytest.fixture
def dagmc_enabled(transport_options):
    if not openmc.lib.feature_enabled('dagmc'):
        pytest.skip('DAGMC CAD geometry is not enabled')
    if not CAD.is_file():
        pytest.skip('DAGMC sphere fixture is unavailable')


def _nickel(density=8.9, name='nickel'):
    material = openmc.Material(name=name)
    material.add_nuclide('Ni64', 1.)
    material.set_density('g/cm3', density)
    return material


def _settings(model, center=(0., 0., 0.), particles=1000):
    model.settings.run_mode = 'fixed source'
    model.settings.batches = 3
    model.settings.particles = particles
    model.settings.seed = 1729
    model.settings.verbosity = 1
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(center),
        energy=openmc.stats.Discrete([1.], [1.]))
    model.settings.output = {'summary': False, 'tallies': False, 'path': '.'}
    model.tallies = [_tally('all photons'), _tally('by parent', True)]


def _sphere(radius=10., center=(0., 0., 0.), density=8.9):
    material = _nickel(density)
    boundary = openmc.Sphere(x0=center[0], y0=center[1], z0=center[2],
                             r=radius, boundary_type='vacuum')
    cell = openmc.Cell(fill=material, region=-boundary)
    model = openmc.Model(geometry=openmc.Geometry([cell]),
                         materials=[material])
    _settings(model, center)
    return model, cell


def _run(model, directory, options):
    with openmc.config.patch('chain_file', CHAIN):
        result = Path(model.run(cwd=directory, **options))
    return result if result.is_absolute() else directory / result


@pytest.fixture(scope='module')
def killed_capture(transport_options, tmp_path_factory):
    """Ordinary two-site FileSource: rejection must kill, not resample."""
    directory = tmp_path_factory.mktemp('d1s-killed-histories')
    source_file = directory / 'neutrons.h5'
    openmc.write_source_file([
        openmc.SourceParticle(E=energy, particle=openmc.ParticleType.NEUTRON)
        for energy in (.1, 1.)], source_file)
    model, cell = _sphere()
    model.settings.particles = 2000
    model.settings.source = openmc.FileSource(source_file, constraints={
        'energy_bounds': (.9, 1.1), 'rejection_strategy': 'kill'})
    model.settings.photon_transport = True
    model.settings.use_decay_photons = True
    model.settings.d1s_source_write = True
    _run(model, directory / 'capture', transport_options)
    files = tuple(sorted((directory / 'capture').glob('d1s_source.*.h5')))
    assert len(files) == 1  # This fixture deliberately uses serial transport.

    ordinary = deepcopy(model)
    ordinary.settings.d1s_source_write = False
    tally = openmc.Tally(name='neutron photon production')
    # The events score uses each produced secondary's weight, not the
    # incident neutron's weight. The neutron filter excludes descendants.
    # Do NOT raise energy_photon: create_secondary applies that cutoff before
    # ParticleProductionFilter can see the birth.
    tally.filters = [openmc.ParticleFilter('neutron'),
                     openmc.ParticleProductionFilter('photon')]
    tally.scores = ['events']
    tally.estimator = 'analog'
    ordinary.tallies = [tally]
    statepoint = _run(ordinary, directory / 'ordinary', transport_options)
    with openmc.StatePoint(statepoint, autolink=False) as sp:
        expected = sp.tallies[tally.id].mean.item()
    config = d1s.Configuration(model, {'nickel': d1s.Cells([cell])})
    return config, files, expected


def test_capture_matches_ordinary_production_per_external_history(
    killed_capture):
    """A direct backend oracle independent of Python's kernel denominator."""
    config, files, expected = killed_capture
    histories = config.model.settings.particles * config.model.settings.batches
    assert histories == 6000
    with h5py.File(files[0]) as handle:
        birth_weight = handle['source_bank']['wgt'].sum()
    assert birth_weight > 0.
    assert expected > 0.
    assert birth_weight / histories == pytest.approx(expected, rel=2.e-12)


def test_capture_counts_zero_weight_killed_source_histories(killed_capture):
    _, files, _ = killed_capture
    with h5py.File(files[0]) as handle:
        count = handle['source_histories']
        assert count.shape == ()
        assert count.dtype.kind == 'i' and count.dtype.itemsize == 8
        assert count[()] == 6000
        weight = handle['source_weight']
        assert weight.shape == ()
        # Both sites have unit weight before rejection. About half the
        # histories survive, so the informational weight is NOT the count.
        assert 2400. < weight[()] < 3600.


def test_kernel_emission_rate_uses_history_count(killed_capture,
                                               transport_options, tmp_path):
    config, files, expected_production = killed_capture
    kernel = _d1s._read_kernel(config, files)
    assert kernel.source_histories == 6000
    assert 2400. < kernel.source_weight < 3600.
    parents = np.asarray(kernel.parent_names)[kernel.parent_nuclide]
    assert set(parents) == {'Ni65'}
    sim = _simulation([d1s.Irradiation(config, 3600., 17.),
                       d1s.Cooling(config, 1., [0.])])
    production = d1s.Production({config: kernel}, files[0].parent, CHAIN)
    result = sim.evaluate(production, tmp_path / 'replay', transport_options)
    expected = expected_production * _factor('Ni65', 3600., 17., 0.)
    assert result.source_rates['cooling-1', 0.] == pytest.approx(
        expected, rel=2.e-12)


def _reversed_csg_placements():
    material = _nickel()
    surface = openmc.Sphere(r=1.)
    leaf = openmc.Cell(fill=material, region=-surface)
    universe = openmc.Universe(cells=[leaf, openmc.Cell(region=+surface)])
    placements = []
    # Insertion order says left first, but exported XML sorts cells by ID.
    # The backend therefore assigns leaf instance 0 to the RIGHT placement.
    for uid, x in ((900, -3.), (800, 3.)):
        boundary = openmc.Sphere(x0=x, r=1.5, boundary_type='vacuum')
        cell = openmc.Cell(cell_id=uid, fill=universe, region=-boundary)
        cell.translation = (x, 0., 0.)
        placements.append(cell)
    model = openmc.Model(geometry=openmc.Geometry(placements),
                         materials=[material])
    _settings(model, (3., 0., 0.))
    return model, leaf


@pytest.mark.parametrize('instance', [0, 1],
                         ids=['right-valid', 'left-invalid'])
def test_csg_replay_uses_xml_instance_order(instance, transport_options,
                                           tmp_path):
    origin_model, origin_cell = _sphere(radius=1., center=(3., 0., 0.))
    target_model, target_cell = _reversed_csg_placements()
    assert list(target_model.geometry.root_universe.cells) == [900, 800]
    origin = d1s.Configuration(origin_model, {
        'part': d1s.Cells([origin_cell])})
    target = d1s.Configuration(target_model, {
        'part': d1s.Cells([target_cell], [instance])})
    sim = _simulation([d1s.Irradiation(origin, 3600., 1.),
                       d1s.Cooling(target, 1., [0.])])
    production = sim.prepare(tmp_path / 'capture', transport_options)
    kernel = production.kernels[origin]
    assert len(kernel.bank) > 0
    points = _vectors(kernel.bank, 'r')
    assert np.all(np.linalg.norm(points - [3., 0., 0.], axis=1) < 1.)
    if instance == 1:
        with pytest.raises(ValueError, match='target cell/instance'):
            sim.evaluate(production, tmp_path / 'wrong', transport_options)
    else:
        result = sim.evaluate(production, tmp_path / 'correct',
                       transport_options)
        assert result.get_tally('all photons', 'cooling-1', 0.).mean.sum() > 0.
    assert list(target_model.geometry.root_universe.cells) == [900, 800]


def _dagmc_model(repeated=False):
    material = _nickel(name='test_mat')
    first = openmc.DAGMCUniverse(CAD.resolve(), auto_geom_ids=True)
    second = (openmc.DAGMCUniverse(CAD.resolve(), auto_geom_ids=True)
              if repeated else None)
    placements = []
    universes = (first, first, second) if repeated else (first,)
    for i, universe in enumerate(universes):
        x = 20. * i
        boundary = openmc.Sphere(x0=x, r=6., boundary_type='vacuum')
        cell = openmc.Cell(cell_id=100 * (i + 1), fill=universe,
                   region=-boundary)
        cell.translation = (x, 0., 0.)
        placements.append(cell)
    model = openmc.Model(geometry=openmc.Geometry(placements),
                         materials=[material])
    _settings(model, (20., 0., 0.) if repeated else (0., 0., 0.))
    model.settings.material_cell_offsets = False
    return model, first, second


@pytest.mark.parametrize('destination', ['second placement', 'first placement',
                                        'other universe'])
def test_repeated_dagmc_real_capture_and_preflight(destination, dagmc_enabled,
                                                  transport_options, tmp_path):
    model, first, second = _dagmc_model(repeated=True)
    assert _d1s._raw_volumes(CAD) == (1,)
    origin = d1s.Configuration(model, {
        'part': d1s.DAGMCVolumes(first, [1], [1])})
    target_universe = second if destination == 'other universe' else first
    instance = 1 if destination == 'second placement' else 0
    target = d1s.Configuration(model, {
        'part': d1s.DAGMCVolumes(target_universe, [1], [instance])})
    sim = _simulation([d1s.Irradiation(origin, 3600., 1.),
                       d1s.Cooling(target, 1., [0.])])
    production = sim.prepare(tmp_path / 'capture', transport_options)
    kernel = production.kernels[origin]
    assert len(kernel.bank) > 0
    assert set(kernel.metadata['universe_id']) == {first.id}
    assert set(kernel.metadata['dagmc_volume_id']) == {1}
    assert set(kernel.metadata['cell_instance']) == {1}
    assert np.all(kernel.metadata['cell_id'] != 1)
    assert np.all(np.linalg.norm(
        _vectors(kernel.bank, 'r') - [20., 0., 0.], axis=1) < 5.)
    if destination != 'second placement':
        # Require the actual locator diagnostic, not just any backend failure.
        with pytest.raises(RuntimeError,
                   match='outside its target component/instance'):
            sim.evaluate(production, tmp_path / 'wrong', transport_options)
    else:
        result = sim.evaluate(production, tmp_path / 'correct',
                       transport_options)
        obs = result.observations['cooling-1', 0.]
        assert obs.provenance['target_validation'] == 'isolated C API'
        assert result.get_tally('all photons', 'cooling-1', 0.).mean.sum() > 0.
    assert model.settings.material_cell_offsets is False


def _prescribed_production(config, directory, energy):
    """Prescribe births; keep transport, composition, and preflight real."""
    directory.mkdir()
    filename = directory / 'd1s_source.0.h5'
    _write_capture(filename, config.model, {
        'bank': _sites(((0., 0., 0.), (0., 0., 0.)),
                       weights=(1., 1.), energies=(energy, energy)),
        'names': PARENTS, 'parents': [0, 1], 'source_weight': 10.})
    # The shared helper predates the count-normalization schema. Keep its
    # module untouched and provide the new required scalar locally.
    with h5py.File(filename, 'a') as handle:
        count = handle.require_dataset('source_histories', shape=(),
                                        dtype=np.int64)
        count[()] = 10
    kernel = _d1s._read_kernel(config, (filename,))
    return d1s.Production({config: kernel}, directory, CHAIN)


def _ancestry_replay(model, energy, directory, options):
    config = d1s.Configuration(model)
    # Saturate both parents so each is sampled often despite disparate lives.
    sim = _simulation([d1s.Irradiation(config, 1.e12, 2.),
                       d1s.Cooling(config, 1., [0.])])
    production = _prescribed_production(config, directory / 'births', energy)
    return sim.evaluate(production, directory / 'replay', options)


def _assert_complete_parent_partition(result, total_name='all photons',
                                      parent_name='by parent'):
    total = result.get_tally(total_name, 'cooling-1', 0.)
    parents = result.get_tally(parent_name, 'cooling-1', 0.)
    assert list(parents.find_filter(openmc.ParentNuclideFilter).bins) == list(
        PARENTS)
    assert np.all(parents.mean > 0.)
    assert total.mean.sum() > 0.
    # These are the SAME histories, not independent Monte Carlo estimates.
    assert parents.mean.sum() == pytest.approx(total.mean.sum(), rel=2.e-12)


def test_parent_partition_survives_photon_weight_window_splitting(
        transport_options, tmp_path):
    model, inner = _sphere(radius=1., density=1.e-20)
    outer_surface = inner.region.surface
    middle = openmc.Sphere(r=.5)
    inner.region = -middle
    outer = openmc.Cell(fill=inner.fill, region=+middle & -outer_surface)
    model.geometry = openmc.Geometry([inner, outer])
    # Put the mesh transition just inside the CSG surface. At the exact mesh
    # boundary the locator can still report the inner bin, missing the drop.
    mesh = openmc.SphericalMesh(r_grid=[0., .49, 1.])
    # A uniformly low window is renormalized at birth and will NOT force
    # splitting. Drop the window across the internal surface instead.
    model.settings.weight_windows = [openmc.WeightWindows(
        mesh, [.5, .01], upper_bound_ratio=2., energy_bounds=[0., 1.e6],
        particle_type='photon', max_split=8)]
    model.settings.weight_windows_on = True
    model.settings.weight_window_checkpoints = {'surface': True,
                                                'collision': True}
    model.settings.max_history_splits = 64
    split = _tally('flux by weight')
    split.filters.append(openmc.WeightFilter([0., .99, 1.01]))
    model.tallies.append(split)
    result = _ancestry_replay(model, .5e6, tmp_path, transport_options)
    by_weight = result.get_tally('flux by weight',
                                 'cooling-1', 0.).mean.ravel()
    assert by_weight[0] > 0.  # Subunit tracks prove that splitting happened.
    assert by_weight[1] > 0.
    total = result.get_tally('all photons', 'cooling-1', 0.)
    assert total.mean.sum() == pytest.approx(
        result.source_rates['cooling-1', 0.], rel=2.e-12)
    _assert_complete_parent_partition(result)


@pytest.mark.parametrize('mechanism,energy', [('photoelectric', 3.e4),
                                            ('bremsstrahlung', 1.e6)])
def test_parent_partition_includes_secondary_photons(
    mechanism, energy, transport_options, tmp_path):
    model, _ = _sphere()
    model.settings.electron_treatment = (
        'ttb' if mechanism == 'bremsstrahlung' else 'led')
    model.settings.atomic_relaxation = mechanism == 'photoelectric'
    model.settings.cutoff = {'energy_photon': 1.e3}
    production = openmc.Tally(name='secondary photon births')
    production.filters = [openmc.ParticleFilter('photon'),
                          openmc.ParticleProductionFilter('photon')]
    production.scores = ['events']
    production.estimator = 'analog'
    model.tallies.append(production)
    # Both energies are below pair threshold. With relaxation disabled, the
    # high-energy case can create new photons only through bremsstrahlung.
    low = _tally('downshifted photons')
    low.filters.append(openmc.EnergyFilter([1.e3, .9 * energy]))
    low_parent = deepcopy(low)
    low_parent.id = None
    low_parent.name = 'downshifted by parent'
    low_parent.filters.append(openmc.ParentNuclideFilter(PARENTS))
    model.tallies.extend([low, low_parent])
    if mechanism == 'photoelectric':
        absorption = _tally('photoelectric events')
        absorption.scores = ['photoelectric']
        absorption.estimator = 'analog'
        model.tallies.append(absorption)
    result = _ancestry_replay(model, energy, tmp_path, transport_options)
    assert result.get_tally('secondary photon births',
                            'cooling-1', 0.).mean.sum() > 0.
    if mechanism == 'photoelectric':
        assert result.get_tally('photoelectric events',
                                'cooling-1', 0.).mean.sum() > 0.
    _assert_complete_parent_partition(result, 'downshifted photons',
                                      'downshifted by parent')
    _assert_complete_parent_partition(result)


def test_zero_dagmc_distribcell_matches_nonzero_runtime_shape(
        dagmc_enabled, transport_options, tmp_path):
    model, _, _ = _dagmc_model()
    # The sole placement is cell 100. auto_geom_ids assigns raw volume 1 the
    # runtime ID 101, which deliberately does not exist in Python's geometry.
    assert 101 not in model.geometry.get_all_cells()
    tally = _tally('runtime distribcell')
    tally.filters.append(openmc.DistribcellFilter(101))
    model.tallies = [tally]
    config = d1s.Configuration(model)
    positive = _simulation([d1s.Irradiation(config, 3600., 1.),
                            d1s.Cooling(config, 1., [0.])]).run(
                                tmp_path / 'positive', transport_options)
    reference = positive.get_tally('runtime distribcell', 'cooling-1', 0.)
    assert reference.mean.shape == (1, 1, 1)
    assert reference.mean.item() > 0.
    zero = _simulation([d1s.Irradiation(config, 3600., 0.),
                        d1s.Cooling(config, 1., [0.])]).run(
                            tmp_path / 'zero', transport_options)
    result = zero.get_tally('runtime distribcell', 'cooling-1', 0.)
    assert result.shape == reference.shape == (1, 1, 1)
    assert result.find_filter(openmc.DistribcellFilter).num_bins == 1
    np.testing.assert_array_equal(result.mean, np.zeros(reference.shape))
    np.testing.assert_array_equal(result.std_dev, np.zeros(reference.shape))
    assert result.num_realizations == 0
    obs = zero.observations['cooling-1', 0.]
    assert obs.source_rate == 0.
    assert obs.statepoint is None and obs.source_bank is None
    assert 101 not in model.geometry.get_all_cells()


def test_unirradiated_none_identity_need_not_appear_in_target(
        transport_options, tmp_path):
    model, cell = _sphere()
    origin = d1s.Configuration(model, {
        'nickel': d1s.Cells([cell]), 'future': None})
    target = d1s.Configuration(model, {'nickel': d1s.Cells([cell])})
    sim = _simulation([d1s.Irradiation(origin, 3600., 1.),
                       d1s.Cooling(target, 1., [0.])])
    assert sim.validate() is None
    result = sim.run(tmp_path / 'future', transport_options)
    kernel = result.production.kernels[origin]
    assert len(kernel.bank) > 0
    assert set(kernel.components) == {'nickel'}
    obs = result.observations['cooling-1', 0.]
    assert obs.source_rate > 0.
    assert result.get_tally('all photons', 'cooling-1', 0.).mean.sum() > 0.
    with h5py.File(obs.source_bank) as handle:
        assert handle['component_names'].asstr()[()].tolist() == ['nickel']


@pytest.fixture
def retained_surface_source(tmp_path):
    """Retain a legacy source identical to the prescribed, tagged births."""
    model, _ = _sphere(radius=1., density=1.e-20)
    source_file = tmp_path / 'ordinary_photons.h5'
    # Ordinary source files have no D1S ancestry. Matching the tagged sites
    # makes an accidentally appended source double the analytic flux response,
    # rather than changing attenuation or the source's spatial distribution.
    openmc.write_source_file([
        openmc.SourceParticle(r=(0., 0., 0.), u=(1., 0., 0.), E=1.e6,
                              particle=openmc.ParticleType.PHOTON)
        for _ in PARENTS], source_file)
    with pytest.warns(FutureWarning, match='surf_source_read'):
        model.settings.surf_source_read = {'path': source_file}
    config = d1s.Configuration(model)
    production = _prescribed_production(config, tmp_path / 'births', 1.e6)
    kernel = production.kernels[config]
    assert kernel.source_histories == 10
    np.testing.assert_array_equal(kernel.bank['wgt'], [1., 1.])
    before = _snapshot(model)
    yield config, production
    assert config.model is model
    assert model.settings.surf_source_read == {'path': source_file}
    assert _snapshot(model) == before


def test_cooling_replay_ignores_retained_surface_source(
        retained_surface_source, transport_options, tmp_path, monkeypatch):
    """Replay must replace every external source, not append a second one."""
    config, production = retained_surface_source
    run = openmc.Model.run
    transport_settings = []

    def record_run(model, *args, **kwargs):
        transport_settings.append(deepcopy(model.settings))
        return run(model, *args, **kwargs)

    # Record the clone, but leave transport and statepoint generation real.
    monkeypatch.setattr(openmc.Model, 'run', record_run)
    sim = _simulation([d1s.Irradiation(config, 1.e12, 2.),
                       d1s.Cooling(config, 1., [0.])])
    result = sim.evaluate(production, tmp_path / 'replay', transport_options)
    observation = result.observations['cooling-1', 0.]
    # Both parents saturate: (1 + 1) / 10 histories * 2 neutrons/s = 0.4/s.
    assert observation.source_rate == pytest.approx(.4, rel=2.e-12)
    with openmc.StatePoint(observation.statepoint, autolink=False) as sp:
        raw = sp.get_tally(name='all photons').mean.item()
    physical = result.get_tally('all photons', 'cooling-1', 0.).mean.item()
    # Every 1 MeV photon travels exactly one cm to the vacuum boundary. A
    # retained surf_source_read appends an ordinary FileSource and yields
    # raw=2, physical=0.8, despite the correct composed photon rate of 0.4/s.
    assert (raw, physical) == pytest.approx((1., .4), rel=2.e-12)
    assert len(transport_settings) == 1
    settings = transport_settings[0]
    assert not settings.surf_source_read
    assert len(settings.source) == 1
    assert isinstance(settings.source[0], openmc.FileSource)
    assert settings.source[0].strength == 1.
    assert Path(settings.source[0].path) == observation.source_bank


def test_geometry_locator_ignores_retained_surface_source(
        retained_surface_source, transport_options, tmp_path):
    """The isolated locator must clear legacy sources on its own clone."""
    config, _ = retained_surface_source
    directory = tmp_path / 'locator'
    directory.mkdir()
    model = _d1s._clone(config)
    before = _snapshot(model)
    counts = _d1s._preflight(model, directory)
    assert counts == {uid: 1 for uid in model.geometry.get_all_cells()}
    assert _snapshot(model) == before
    # Inspect the input actually consumed by the real isolated C API process.
    # Plot initialization may not sample this source, but must not retain it.
    locator = ET.parse(str(directory / 'model.xml'))
    assert locator.find('settings/surf_source_read') is None


def test_zero_unstructured_mesh_matches_nonzero_runtime_shape(
        transport_options, tmp_path):
    """An unloaded unstructured mesh must not become an empty zero tally."""
    if not openmc.lib.feature_enabled('dagmc'):
        pytest.skip('DAGMC (and MOAB) mesh support is not enabled')
    mesh_file = (Path(__file__).resolve().parents[1] / 'regression_tests' /
                 'external_moab' / 'test_mesh_tets.h5m')
    if not mesh_file.is_file():
        pytest.skip('MOAB tetrahedral mesh fixture is unavailable')
    model, _ = _sphere(radius=10., density=1.e-20)
    mesh = openmc.UnstructuredMesh(mesh_file, 'moab')
    tally = _tally('unstructured photon flux')
    tally.filters.append(openmc.MeshFilter(mesh))
    model.tallies = [tally]
    config = d1s.Configuration(model)
    before = _snapshot(model)
    production = _prescribed_production(config, tmp_path / 'births', 1.e6)
    positive = _simulation([d1s.Irradiation(config, 1.e12, 2.),
                            d1s.Cooling(config, 1., [0.])]).evaluate(
                                production, tmp_path / 'positive',
                                transport_options)
    reference = positive.get_tally('unstructured photon flux', 'cooling-1', 0.)
    with openmc.StatePoint(positive.statepoint_paths['cooling-1', 0.],
                          autolink=False) as sp:
        n_elements = sp.meshes[mesh.id].n_elements
    assert n_elements == 12000
    expected_shape = (n_elements, 1, 1)
    assert reference.mean.shape == expected_shape
    assert reference.mean.sum() > 0.
    # Do not prime the caller's mesh with statepoint data: the zero path must
    # discover the backend shape independently, without mutating that mesh.
    assert not mesh.has_statepoint_data
    assert _snapshot(model) == before
    zero = _simulation([d1s.Irradiation(config, 1.e12, 0.),
                        d1s.Cooling(config, 1., [0.])]).evaluate(
                            production, tmp_path / 'zero', transport_options)
    result = zero.get_tally('unstructured photon flux', 'cooling-1', 0.)
    assert _snapshot(model) == before
    assert not mesh.has_statepoint_data
    assert result.shape == reference.shape == expected_shape
    assert result.mean.shape == result.std_dev.shape == expected_shape
    assert result.find_filter(openmc.MeshFilter).num_bins == n_elements
    np.testing.assert_array_equal(result.mean, np.zeros(expected_shape))
    np.testing.assert_array_equal(result.std_dev, np.zeros(expected_shape))
    dataframe = result.get_pandas_dataframe()
    assert len(dataframe) == n_elements
    assert set(dataframe['nuclide']) == {'total'}
    assert result.num_realizations == 0
    observation = zero.observations['cooling-1', 0.]
    assert observation.source_rate == 0.
    assert observation.statepoint is None and observation.source_bank is None