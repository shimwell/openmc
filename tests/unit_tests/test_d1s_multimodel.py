"""Multi-model D1S contracts, with synthetic banks and small transport checks.

Synthetic runs replace only Model.run: capture/replay HDF5 I/O, geometry
validation, decay constants, composition, and statepoint reading remain real.
The transport tests use analytic identities, not library-specific references.
"""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import pickle
import shutil

import h5py
import lxml.etree as ET
import numpy as np
import pytest

import openmc
import openmc.deplete
import openmc.lib
from openmc.deplete import d1s, _d1s
from openmc.statepoint import _VERSION_STATEPOINT


CHAIN = (Path(__file__).parents[1] / 'chain_ni.xml').resolve()
CAD = Path(__file__).parent / 'dagmc' / 'dagmc_sphere_r5.h5m'
PARENTS = ('Co60', 'Ni65')
# Minor version 2 uses PDG particle codes (photon=22), not legacy photon=1.
SOURCE_VERSION = (_VERSION_STATEPOINT, 2)
XYZ = np.dtype([(axis, '<f8') for axis in 'xyz'])
SITE = np.dtype([
    ('r', XYZ), ('u', XYZ), ('E', '<f8'), ('time', '<f8'),
    ('wgt', '<f8'), ('delayed_group', '<i4'), ('surf_id', '<i4'),
    ('particle', '<i4'),
])
METADATA = ('cell_id', 'cell_instance', 'universe_id', 'dagmc_volume_id',
            'history_id', 'batch')


def _vectors(bank, key):
    return np.column_stack([bank[key][axis] for axis in 'xyz'])


def _sites(points=((0., 0., 0.), (1., 0., 0.)), weights=(2., 7.),
           energies=(2.e5, 1.e6), directions=None):
    bank = np.zeros(len(points), dtype=SITE)
    for key, values in (('r', points), ('u', directions if directions is not None
                                     else [(1., 0., 0.)] * len(points))):
        for i, axis in enumerate('xyz'):
            bank[key][axis] = np.asarray(values)[:, i]
    bank['wgt'] = weights
    bank['E'] = energies
    bank['particle'] = int(openmc.ParticleType.PHOTON)
    bank['surf_id'] = 12345
    return bank


def _tally(name, parents=False):
    tally = openmc.Tally(name=name)
    tally.filters = [openmc.ParticleFilter('photon')]
    if parents:
        tally.filters.append(openmc.ParentNuclideFilter(PARENTS))
    tally.scores = ['flux']
    return tally


def _snapshot(model):
    """Serialize state without asking lxml elements to support pickling."""
    seen = {}

    def freeze(value):
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return value
        if isinstance(value, type):
            # CheckedList stores its element class. Its global ID registry is
            # not caller-owned model state and legitimately changes on copy.
            return ('type', value.__module__, value.__qualname__)
        if isinstance(value, Path):
            return ('path', str(value))
        if isinstance(value, np.ndarray):
            return ('array', value.dtype.descr, value.shape, value.tobytes())
        if isinstance(value, ET._Element):
            return ('xml', ET.tostring(value))
        if id(value) in seen:
            return ('reference', seen[id(value)])
        seen[id(value)] = len(seen)
        if isinstance(value, dict):
            return tuple((freeze(k), freeze(v)) for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return (type(value).__name__, tuple(map(freeze, value)),
                    freeze(vars(value)) if hasattr(value, '__dict__') else None)
        if isinstance(value, set):
            return ('set', tuple(sorted(map(repr, value))))
        if hasattr(value, '__dict__'):
            return (type(value).__qualname__, freeze(vars(value)))
        return (type(value).__qualname__, repr(value))

    return pickle.dumps(freeze(model), protocol=5)


@pytest.fixture
def unchanged():
    """Check complete caller-owned object state, even after expected failures."""
    snapshots = []
    cwd = Path.cwd()
    config = dict(openmc.config)
    initialized = openmc.lib.is_initialized

    def watch(model):
        snapshots.append((model, _snapshot(model),
                          model.settings, model.geometry, model.tallies,
                          tuple(model.tallies)))
        return model

    yield watch
    for model, serialized, settings, geometry, tallies, entries in snapshots:
        assert model.settings is settings
        assert model.geometry is geometry
        assert model.tallies is tallies
        assert all(a is b for a, b in zip(model.tallies, entries))
        assert _snapshot(model) == serialized
    assert Path.cwd() == cwd
    assert dict(openmc.config) == config
    assert openmc.lib.is_initialized == initialized


@pytest.fixture
def make_model(unchanged):
    def make(center=(0., 0., 0.), radius=10., energy=14.e6, strength=1.,
             material_name='nickel'):
        mat = openmc.Material(name=material_name)
        mat.add_element('Ni', 1.)
        mat.set_density('g/cm3', 8.9)
        sphere = openmc.Sphere(x0=center[0], y0=center[1], z0=center[2],
                               r=radius, boundary_type='vacuum')
        cell = openmc.Cell(fill=mat, region=-sphere)
        model = openmc.Model(geometry=openmc.Geometry([cell]), materials=[mat])
        model.settings.run_mode = 'fixed source'
        model.settings.batches = 3
        model.settings.particles = 1000
        model.settings.verbosity = 1
        model.settings.source = openmc.IndependentSource(
            space=openmc.stats.Point(center),
            energy=openmc.stats.Discrete([energy], [1.]), strength=strength)
        model.settings.output = {'summary': False, 'tallies': False}
        model.tallies = [_tally('all photons'), _tally('by parent', True)]
        return unchanged(model), cell
    return make


def _write_capture(path, model, spec):
    """Write the public on-disk C++ capture schema, including empty ranks."""
    bank = spec.get('bank', _sites())
    cell = next(iter(model.geometry.get_all_cells().values()), None)
    defaults = {
        'cell_id': 0 if cell is None else cell.id,
        'cell_instance': 0, 'universe_id': model.geometry.root_universe.id,
        'dagmc_volume_id': 0, 'history_id': 1, 'batch': 1,
    }
    with h5py.File(path, 'x') as handle:
        handle.attrs['filetype'] = np.bytes_('source')
        handle.attrs['version'] = SOURCE_VERSION
        handle.attrs['d1s_version'] = 1
        handle['source_bank'] = bank
        handle['source_weight'] = spec.get('source_weight', 10.)
        handle['source_histories'] = np.int64(spec.get(
            'source_histories', spec.get('source_weight', 10.)))
        handle.create_dataset('parent_names', data=spec.get('names', PARENTS),
                              dtype=h5py.string_dtype())
        handle['parent_nuclide'] = np.asarray(
            spec.get('parents', np.arange(len(bank)) % 2), dtype=np.int64)
        for key in METADATA:
            handle[key] = np.broadcast_to(
                spec.get(key, defaults[key]), (len(bank),)).astype(np.int64)


def _write_statepoint(path, model):
    """Minimal genuine StatePoint schema with nonzero, unequal batch moments.

    Deliberately prescribed per-source moments isolate Python postprocessing
    from backend source-strength normalization, which is tested in transport.
    """
    with h5py.File(path, 'x') as handle:
        handle.attrs['filetype'] = np.bytes_('statepoint')
        handle.attrs['version'] = SOURCE_VERSION
        handle.attrs['tallies_present'] = 1
        handle['runtime/simulation'] = 1.
        group = handle.create_group('tallies')
        group.attrs['n_tallies'] = len(model.tallies)
        group.attrs['ids'] = [t.id for t in model.tallies]
        group.create_group('meshes')
        filters = group.create_group('filters')
        for tally in model.tallies:
            for filt in tally.filters:
                key = f'filter {filt.id}'
                if key in filters:
                    continue
                fg = filters.create_group(key)
                fg['type'] = np.bytes_(filt.short_name.lower())
                bins = filt.bins
                if isinstance(filt, openmc.ParticleFilter):
                    bins = np.asarray(bins, dtype='S')
                fg['bins'] = bins
                fg['n_bins'] = filt.num_bins
            tg = group.create_group(f'tally {tally.id}')
            tg['name'] = np.bytes_(tally.name)
            tg['estimator'] = np.bytes_(tally.estimator or 'tracklength')
            tg['n_realizations'] = 3
            tg['n_filters'] = len(tally.filters)
            tg['filters'] = [f.id for f in tally.filters]
            tg['nuclides'] = np.asarray(tally.nuclides or ['total'], dtype='S')
            tg['score_bins'] = np.asarray(tally.scores, dtype='S')
            tg.attrs['higher_moments'] = tally.higher_moments
            n = tally.num_filter_bins * len(tally.nuclides or ['total'])
            base = np.arange(1, n * len(tally.scores) + 1).reshape(
                tally.num_filter_bins, -1)
            # Batch scores base*[1, 2, 3] have mean 2*base.
            powers = range(1, 5 if tally.higher_moments else 3)
            tg['results'] = np.stack([
                base**p * sum(x**p for x in (1., 2., 3.))
                for p in powers], axis=-1)


@pytest.fixture
def synthetic(monkeypatch):
    """Queue capture shards; keep _transport and all file readers unmocked."""
    plans, calls = [], []

    def run(model, cwd, **options):
        directory = Path(cwd)
        capture = model.settings.d1s_source_write
        calls.append((capture, directory, model, deepcopy(options)))
        assert Path(openmc.config['chain_file']) == CHAIN
        assert model.settings.photon_transport
        assert model.settings.use_decay_photons == capture
        assert model.settings.output['path'] == '.'
        if capture:
            shards = plans.pop(0) if plans else [{}]
            for rank, spec in enumerate(shards):
                _write_capture(directory / f'd1s_source.{rank}.h5', model, spec)
        else:
            assert isinstance(model.settings.source[0], openmc.FileSource)
            assert not model.settings.uniform_source_sampling
        path = directory / 'statepoint.3.h5'
        _write_statepoint(path, model)
        return path

    monkeypatch.setattr(openmc.Model, 'run', run)
    return plans, calls


def _simulation(steps):
    return d1s.Simulation(steps, chain_file=CHAIN)


def _factor(parent, duration, rate, age):
    lam = openmc.data.decay_constant(parent, chain_file=CHAIN)
    return rate * -np.expm1(-lam * duration) * np.exp(-lam * age)


def _digests(directory):
    return {p.relative_to(directory): sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


@pytest.mark.parametrize('value', [-1., 0., np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('kind', [d1s.Irradiation, d1s.Cooling])
def test_duration_validation(make_model, value, kind):
    model, _ = make_model()
    configuration = d1s.Configuration(model)
    with pytest.raises(ValueError, match='positive|nonnegative'):
        if kind is d1s.Irradiation:
            kind(configuration, value, 1.)
        else:
            kind(configuration, value)


@pytest.mark.parametrize('value', [True, '1', object()])
def test_duration_type_validation(make_model, value):
    model, _ = make_model()
    with pytest.raises(TypeError):
        d1s.Cooling(d1s.Configuration(model), value)


@pytest.mark.parametrize('value', [-1., np.nan, np.inf, True, '1'])
def test_source_rate_validation(make_model, value):
    model, _ = make_model()
    with pytest.raises((TypeError, ValueError)):
        d1s.Irradiation(d1s.Configuration(model), 1., value)


@pytest.mark.parametrize('times', [[-1.], [11.], [1., (1., 's')], [np.nan],
                                 [True], [(1., 'fortnight')]])
def test_sample_time_validation(make_model, times):
    model, _ = make_model()
    with pytest.raises((TypeError, ValueError)):
        d1s.Cooling(d1s.Configuration(model), 10., times)


def test_units_names_and_immutable_maps(make_model):
    model, cell = make_model()
    mapping = {'part': d1s.Cells([cell])}
    transform = np.eye(4)
    configuration = d1s.Configuration(model, mapping, {'part': transform})
    mapping.clear()
    transform[0, 3] = 42.
    assert list(configuration.components) == ['part']
    assert configuration.transforms['part'][0, 3] == 0.
    with pytest.raises(TypeError):
        configuration.components['new'] = None
    with pytest.raises(ValueError):
        configuration.transforms['part'][0, 3] = 4.
    irradiation = d1s.Irradiation(configuration, (2., 'min'), 0.)
    cooling = d1s.Cooling(configuration, (1., 'h'),
                          [(1., 'min'), 0., (1., 'h')])
    assert irradiation.duration == 120.
    assert cooling.sample_times == (0., 60., 3600.)
    assert _simulation([irradiation, cooling]).step_names == (
        'irradiation-1', 'cooling-1')
    assert d1s.Cooling(configuration, 1.).sample_times == ()


@pytest.mark.parametrize('matrix', [np.eye(3), np.full((4, 4), np.nan),
    np.diag([2., 1., 1., 1.]), np.diag([-1., 1., 1., 1.]),
    np.diag([1., 1., 1., 2.]), np.eye(4, dtype=complex) + 1.j])
def test_invalid_rigid_transforms(make_model, matrix):
    model, cell = make_model()
    with pytest.raises(ValueError):
        d1s.Configuration(model, {'part': d1s.Cells([cell])}, {'part': matrix})


@pytest.mark.parametrize('case', ['model', 'selector', 'empty identity',
                                'unknown transform', 'unmapped transform'])
def test_configuration_validation(make_model, case):
    model, cell = make_model()
    with pytest.raises((TypeError, ValueError)):
        if case == 'model':
            d1s.Configuration(object())
        elif case == 'selector':
            d1s.Configuration(model, {'part': cell})
        elif case == 'empty identity':
            d1s.Configuration(model, {' ': d1s.Cells([cell])})
        elif case == 'unknown transform':
            d1s.Configuration(model, {}, {'unknown': np.eye(4)})
        else:
            d1s.Configuration(model, transforms={'part': np.eye(4)})


@pytest.mark.parametrize('values,instances', [([], None), ([0], None),
    ([True], None), ([1, 1], None), ([1, 1], [None, 0]),
    ([1, 1], [0, 0]), ([1], [-1]), ([1], [True]), ([1, 2], [0])])
def test_invalid_selectors(values, instances):
    with pytest.raises((TypeError, ValueError)):
        d1s.Cells(values, instances)
    universe = openmc.DAGMCUniverse(CAD)
    with pytest.raises((TypeError, ValueError)):
        d1s.DAGMCVolumes(universe, values, instances)


def test_nonleaf_selectors(make_model):
    model, cell = make_model()
    for invalid in (openmc.Cell(), openmc.Cell(fill=openmc.Universe(cells=[cell])),
                    openmc.DAGMCCell(900001)):
        with pytest.raises(ValueError, match='CSG leaf'):
            d1s.Cells([invalid])
    with pytest.raises(TypeError):
        d1s.DAGMCVolumes(model.geometry.root_universe, [1])


@pytest.mark.parametrize('case', ['empty', 'step type', 'duplicate',
                                'generated collision', 'missing identity',
                                'mixed maps'])
def test_schedule_validation(make_model, case):
    model, cell = make_model()
    config = d1s.Configuration(model, {'old': d1s.Cells([cell])})
    irradiation = d1s.Irradiation(config, 1., 1.)
    cases = {
        'empty': [], 'step type': [model],
        'duplicate': [d1s.Cooling(config, 1., name='same')] * 2,
        'generated collision': [irradiation,
                                d1s.Cooling(config, 1., name='irradiation-1')],
        'missing identity': [irradiation, d1s.Cooling(
            d1s.Configuration(model, {'replacement': d1s.Cells([cell])}), 1.)],
        'mixed maps': [irradiation, d1s.Cooling(d1s.Configuration(model), 1.)],
    }
    with pytest.raises((TypeError, ValueError)):
        _simulation(cases[case])


@pytest.mark.parametrize('case', ['eigenvalue', 'multigroup', 'random ray',
                                'missing cell', 'foreign cell', 'instance',
                                'overlap'])
def test_validate_transport_and_membership(make_model, unchanged, case):
    # Construct malformed settings before taking the immutability snapshot.
    model, cell = make_model()
    model = deepcopy(model)
    cell = model.geometry.get_all_cells()[cell.id]
    selectors = {'part': d1s.Cells([cell])}
    if case == 'eigenvalue':
        model.settings.run_mode = 'eigenvalue'
    elif case == 'multigroup':
        model.settings.energy_mode = 'multi-group'
    elif case == 'random ray':
        model.settings.random_ray = {'distance_active': 1.}
    elif case == 'missing cell':
        selectors = {'part': d1s.Cells([999999])}
    elif case == 'foreign cell':
        selectors = {'part': d1s.Cells([deepcopy(cell)])}
    elif case == 'instance':
        selectors = {'part': d1s.Cells([cell], [1])}
    else:
        selectors['other'] = d1s.Cells([cell], [0])
    unchanged(model)
    with pytest.raises(ValueError):
        _simulation([d1s.Cooling(d1s.Configuration(model, selectors), 1.)]).validate()


def test_chain_and_run_option_validation(make_model, tmp_path, synthetic,
                                         monkeypatch):
    model, _ = make_model()
    config = d1s.Configuration(model)
    sim = _simulation([d1s.Irradiation(config, 1., 1.)])
    for key in ('cwd', 'restart_file', 'export_model_xml', 'apply_tally_results'):
        with pytest.raises(ValueError, match='run_kwargs'):
            sim.prepare(tmp_path / key, {key: 'forbidden'})
        assert not (tmp_path / key).exists()
    missing = d1s.Simulation(sim.steps, chain_file=tmp_path / 'missing.xml')
    with pytest.raises(FileNotFoundError):
        missing.validate()
    with openmc.config.patch('chain_file', CHAIN):
        assert d1s.Simulation(sim.steps).chain_file == CHAIN
    # Config paths cannot be assigned None; delete and restore through its API.
    previous = openmc.config.get('chain_file')
    try:
        if previous is not None:
            del openmc.config['chain_file']
        no_chain = d1s.Simulation(sim.steps)
    finally:
        if previous is not None:
            openmc.config['chain_file'] = previous
    with pytest.raises(ValueError, match='requires chain_file'):
        no_chain.prepare(tmp_path / 'no-chain')
    with monkeypatch.context() as patch:
        patch.setattr(openmc.Model, 'is_initialized', property(lambda self: True))
        with pytest.raises(RuntimeError, match='finalize it explicitly'):
            sim.validate()


def test_two_kernels_cumulative_ages_and_distinct_banks(make_model, synthetic,
                                                      tmp_path):
    first, a = make_model(energy=1.e6)
    second, b = make_model(energy=14.e6)
    target, c = make_model(radius=20.)
    configs = [d1s.Configuration(m, {'part': d1s.Cells([cell])})
               for m, cell in ((first, a), (second, b), (target, c))]
    bank_a = _sites(points=((-2., 0., 0.), (-1., 0., 0.)))
    bank_b = _sites(points=((3., 0., 0.), (4., 0., 0.)),
                    weights=(11., 3.), energies=(3.e6, 5.e6))
    plans, calls = synthetic
    plans.extend([[{'bank': bank_a, 'source_weight': 20.}],
                  [{'bank': bank_b, 'parents': [1, 0], 'source_weight': 40.}]])
    steps = [d1s.Irradiation(configs[0], 120., 13., name='first'),
             d1s.Irradiation(configs[1], 240., 29., name='second'),
             d1s.Cooling(configs[2], 180., [180., 0., 60.], name='observe')]
    sim = _simulation(steps)
    production = sim.prepare(tmp_path / 'capture', {'threads': 2})
    before = _digests(production.output_dir)
    result = sim.evaluate(production, tmp_path / 'evaluation')
    assert len(production.kernels) == 2
    assert [capture for capture, *_ in calls] == [True, True, False, False, False]
    assert all(model is not original for _, _, model, _ in calls
               for original in (first, second, target))
    assert _digests(production.output_dir) == before
    assert len(set(result.statepoint_paths.values())) == 3
    for local in (0., 60., 180.):
        obs = result.observations['observe', local]
        assert obs.step_index == 2
        assert obs.global_time == 360. + local
        expected = np.array([
            2. / 20. * _factor('Co60', 120., 13., 240. + local),
            7. / 20. * _factor('Ni65', 120., 13., 240. + local),
            11. / 40. * _factor('Ni65', 240., 29., local),
            3. / 40. * _factor('Co60', 240., 29., local)])
        assert obs.source_rate == pytest.approx(expected.sum())
        with h5py.File(obs.source_bank) as handle:
            np.testing.assert_allclose(handle['sampling_weights'], expected)
            replay = handle['source_bank'][()]
            np.testing.assert_array_equal(replay['r'],
                                          np.concatenate([bank_a, bank_b])['r'])
            np.testing.assert_array_equal(replay['E'], [2.e5, 1.e6, 3.e6, 5.e6])
            np.testing.assert_array_equal(handle['kernel_index'], [0, 0, 1, 1])
            names = handle['parent_names'].asstr()[()]
            np.testing.assert_array_equal(names[handle['parent_nuclide'][()]],
                                          ['Co60', 'Ni65', 'Ni65', 'Co60'])
            assert np.all(replay['wgt'] == 1.)
            assert np.all(replay['surf_id'] == 0)
        assert len(obs.provenance['kernels']) == 2
        assert 'neutron bank generation' in result.provenance['uncertainty']


def test_same_kernel_time_correction_and_production_reuse(make_model, synthetic,
                                                        tmp_path):
    model, _ = make_model()
    config = d1s.Configuration(model)
    schedule = _simulation([
        d1s.Irradiation(config, 100., 3.),
        d1s.Cooling(config, 20., [0., 20.], name='early'),
        d1s.Irradiation(config, 80., 7.),
        d1s.Cooling(config, 40., [0., 40.], name='late')])
    plans, calls = synthetic
    # Rank one has no photons but contributes half of the neutron normalization.
    plans.append([{'source_weight': 10.},
                  {'bank': np.zeros(0, dtype=SITE), 'source_weight': 10.}])
    production = schedule.prepare(tmp_path / 'capture')
    assert len(production.kernels) == 1
    kernel = production.kernels[config]
    assert kernel.source_weight == 20.
    for array in (kernel.bank, kernel.parent_nuclide, kernel.components,
                  *kernel.metadata.values()):
        assert not array.flags.writeable
    result = schedule.evaluate(production, tmp_path / 'evaluate')
    for step, local, intervals, rates in (
            ('early', 0., [100.], [3.]),
            ('early', 20., [100., 20.], [3., 0.]),
            ('late', 0., [100., 20., 80.], [3., 0., 7.]),
            ('late', 40., [100., 20., 80., 40.], [3., 0., 7., 0.])):
        factors = d1s.time_correction_factors(PARENTS, intervals, rates)
        expected = np.array([2., 7.]) / 20. * [factors[p][-1] for p in PARENTS]
        with h5py.File(result.observations[step, local].source_bank) as handle:
            # The legacy helper uses bundled half-lives rather than the chain;
            # their slight rounding differences need a relative tolerance.
            np.testing.assert_allclose(handle['sampling_weights'], expected,
                                       rtol=2.e-5)
    # Different rates/durations and cooling geometry must not recapture.
    target, _ = make_model(radius=15.)
    alternate = _simulation([d1s.Irradiation(config, 30., 100.),
        d1s.Cooling(d1s.Configuration(target), 1., [0.])])
    alternate.evaluate(production, tmp_path / 'reuse')
    assert sum(capture for capture, *_ in calls) == 1
    different = _simulation([d1s.Irradiation(d1s.Configuration(model), 1., 1.)])
    with pytest.raises(ValueError, match='same objects'):
        different.evaluate(production, tmp_path / 'wrong-identity')


@pytest.mark.parametrize('case', ['zero rate', 'empty bank', 'stable parent',
                                'removed', 'cooling only'])
def test_zero_source_skips_transport(make_model, synthetic, unchanged, tmp_path,
                                     case):
    model, cell = make_model()
    model = deepcopy(model)
    tally = model.tallies[0]
    tally.higher_moments = True
    unchanged(model)
    cell = model.geometry.get_all_cells()[cell.id]
    config = d1s.Configuration(model, {'part': d1s.Cells([cell])})
    target = (d1s.Configuration(model, {'part': None})
              if case == 'removed' else config)
    plans, calls = synthetic
    spec = {}
    if case == 'empty bank':
        spec['bank'] = np.zeros(0, dtype=SITE)
    elif case == 'stable parent':
        spec.update(names=['Ni58'], parents=[0, 0])
    plans.append([spec])
    steps = [] if case == 'cooling only' else [d1s.Irradiation(
        config, 100., 0. if case == 'zero rate' else 1.)]
    steps.append(d1s.Cooling(target, 10., [0., 10.]))
    result = _simulation(steps).run(tmp_path / 'zero')
    assert not any(not capture for capture, *_ in calls)
    for obs in result.observations.values():
        assert obs.source_rate == 0.
        assert obs.statepoint is None and obs.source_bank is None
        zero = obs.tallies[tally.id]
        assert zero.shape == (1, 1, 1)
        np.testing.assert_array_equal(zero.mean, np.zeros(zero.shape))
        np.testing.assert_array_equal(zero.std_dev, np.zeros(zero.shape))
        assert zero.num_realizations == 0
        assert not zero._sum_third.any() and not zero._sum_fourth.any()


def test_removal_reinsertion_and_replacement_identity(make_model, synthetic,
                                                    tmp_path):
    model, cell = make_model()
    replacement, new_cell = make_model(material_name='nickel')
    original = d1s.Configuration(model, {'old': d1s.Cells([cell]), 'new': None})
    removed = d1s.Configuration(replacement, {'old': None,
                                            'new': d1s.Cells([new_cell])})
    restored = d1s.Configuration(model, {'old': d1s.Cells([cell]), 'new': None})
    plans, _ = synthetic
    plans.extend([[{}], [{'bank': _sites(weights=(19., 23.))}]])
    result = _simulation([
        d1s.Irradiation(original, 100., 5.),
        d1s.Cooling(removed, 20., [0.], name='removed'),
        d1s.Irradiation(removed, 200., 11.),
        d1s.Cooling(removed, 30., [0.], name='replacement'),
        d1s.Cooling(restored, 10., [0.], name='returned'),
    ]).run(tmp_path / 'lifecycle')
    assert result.source_rates['removed', 0.] == 0.
    for step, weights, duration, rate, age, component in (
            ('replacement', [19., 23.], 200., 11., 0., 'new'),
            ('returned', [2., 7.], 100., 5., 250., 'old')):
        obs = result.observations[step, 0.]
        expected = np.array(weights) / 10. * [
            _factor(parent, duration, rate, age) for parent in PARENTS]
        assert obs.source_rate == pytest.approx(expected.sum())
        with h5py.File(obs.source_bank) as handle:
            np.testing.assert_allclose(handle['sampling_weights'], expected)
            names = handle['component_names'].asstr()[()]
            assert set(names[handle['target_component'][()]]) == {component}


def _rigid(angle, translation):
    radians = np.radians(angle)
    c, s = np.cos(radians), np.sin(radians)
    result = np.eye(4)
    result[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
    result[:3, 3] = translation
    return result


def test_positions_and_directions_use_relative_rigid_transform(make_model,
                                                              synthetic,
                                                              tmp_path):
    origin, a = make_model(center=(3., 4., 0.))
    target, b = make_model(center=(-5., 2., 1.))
    origin_matrix = _rigid(35., (3., 4., 0.))
    target_matrix = _rigid(125., (-5., 2., 1.))
    reference = np.array([[1., 2., 0.], [-1., 0., 1.]])
    directions = np.array([[1., 0., 0.], [0., 1., 0.]])
    bank = _sites(reference @ origin_matrix[:3, :3].T + origin_matrix[:3, 3],
                  directions=directions @ origin_matrix[:3, :3].T)
    configs = [d1s.Configuration(m, {'part': d1s.Cells([cell])}, {'part': matrix})
               for m, cell, matrix in ((origin, a, origin_matrix),
                                        (target, b, target_matrix))]
    plans, _ = synthetic
    plans.append([{'bank': bank, 'cell_instance': [0, 0],
                   'history_id': [7, 19], 'batch': [1, 3]}])
    result = _simulation([d1s.Irradiation(configs[0], 60., 1.),
        d1s.Cooling(configs[1], 1., [0.])]).run(tmp_path / 'transform')
    obs = result.observations['cooling-1', 0.]
    with h5py.File(obs.source_bank) as handle:
        replay = handle['source_bank'][()]
        np.testing.assert_allclose(_vectors(replay, 'r'),
            reference @ target_matrix[:3, :3].T + target_matrix[:3, 3])
        np.testing.assert_allclose(_vectors(replay, 'u'),
            directions @ target_matrix[:3, :3].T, atol=1.e-15)
        np.testing.assert_allclose(np.linalg.norm(_vectors(replay, 'u'), axis=1), 1.)
        np.testing.assert_array_equal(handle['history_id'], [7, 19])
        np.testing.assert_array_equal(handle['batch'], [1, 3])
        np.testing.assert_array_equal(handle['cell_id'], [a.id, a.id])
    np.testing.assert_array_equal(
        result.production.kernels[configs[0]].bank, bank)


def _repeated_model():
    material = openmc.Material()
    material.add_element('Ni', 1.)
    leaf = openmc.Cell(fill=material, region=-openmc.Sphere(r=1.))
    universe = openmc.Universe(cells=[leaf])
    placements = []
    for x in (-3., 3.):
        cell = openmc.Cell(fill=universe, region=-openmc.Sphere(x0=x, r=1.))
        cell.translation = (x, 0., 0.)
        placements.append(cell)
    model = openmc.Model(geometry=openmc.Geometry(placements), materials=[material])
    model.settings.run_mode = 'fixed source'
    model.settings.batches = 3
    model.settings.particles = 10
    model.tallies = [_tally('instances')]
    return model, leaf


def test_instance_origin_membership_and_target_membership(unchanged, synthetic,
                                                         tmp_path):
    model, leaf = _repeated_model()
    unchanged(model)
    origin = d1s.Configuration(model, {'left': d1s.Cells([leaf], [0]),
                                     'right': d1s.Cells([leaf], [1])})
    target = d1s.Configuration(model, {'left': d1s.Cells([leaf], [1]),
                                      'right': None},
                               {'left': _rigid(0., (6., 0., 0.))})
    plans, _ = synthetic
    plans.append([{'bank': _sites(((-3., 0., 0.), (3., 0., 0.))),
                   'cell_id': leaf.id, 'cell_instance': [0, 1]}])
    sim = _simulation([d1s.Irradiation(origin, 100., 1.),
                        d1s.Cooling(target, 1., [0.])])
    production = sim.prepare(tmp_path / 'capture')
    result = sim.evaluate(production, tmp_path / 'correct-instance')
    with h5py.File(result.observations['cooling-1', 0.].source_bank) as handle:
        assert len(handle['source_bank']) == 1
        np.testing.assert_allclose(_vectors(handle['source_bank'][()], 'r'),
                                   [[3., 0., 0.]])
        np.testing.assert_array_equal(handle['cell_instance'], [0])
    wrong = d1s.Configuration(model, {'left': d1s.Cells([leaf], [0]), 'right': None},
                              {'left': _rigid(0., (6., 0., 0.))})
    with pytest.raises(ValueError, match='target cell/instance'):
        _simulation([sim.steps[0], d1s.Cooling(wrong, 1., [0.])]).evaluate(
            production, tmp_path / 'wrong-instance')


@pytest.mark.parametrize('case', ['unmapped capture', 'outside target',
                                'transport failure', 'replay failure',
                                'missing shards'])
def test_failures_preserve_models_and_prior_outputs(make_model, synthetic,
                                                   monkeypatch, tmp_path, case):
    model, cell = make_model()
    target, other = make_model(center=(50., 0., 0.))
    config = d1s.Configuration(model, {'part': d1s.Cells([cell])})
    destination = d1s.Configuration(target, {'part': d1s.Cells([other])})
    sim = _simulation([d1s.Irradiation(config, 100., 1.),
                        d1s.Cooling(destination, 1., [0.])])
    plans, _ = synthetic
    sentinel = tmp_path / 'caller-output'
    sentinel.mkdir()
    marker = sentinel / 'keep.h5'
    marker.touch()
    before = _digests(sentinel)
    if case == 'unmapped capture':
        plans.append([{'cell_id': cell.id + 100000}])
        with pytest.raises(ValueError, match='exactly one component'):
            sim.prepare(tmp_path / 'bad-capture')
    elif case == 'missing shards':
        with pytest.raises(RuntimeError, match='Missing D1S MPI shards'):
            sim.prepare(tmp_path / 'shards', {'mpi_args': ['mpiexec', '-n', '2']})
    elif case in ('transport failure', 'replay failure'):
        production = None
        if case == 'replay failure':
            sim = _simulation([sim.steps[0], d1s.Cooling(config, 1., [0.])])
            production = sim.prepare(tmp_path / 'capture')
            capture_before = _digests(production.output_dir)

        def fail(model, cwd, **kwargs):
            model.settings.source = openmc.IndependentSource(strength=42.)
            raise RuntimeError('synthetic transport failure')
        monkeypatch.setattr(openmc.Model, 'run', fail)
        with pytest.raises(RuntimeError, match='synthetic transport failure'):
            if production is None:
                sim.prepare(tmp_path / 'failure')
            else:
                sim.evaluate(production, tmp_path / 'failure')
        if production is not None:
            assert _digests(production.output_dir) == capture_before
    else:
        production = sim.prepare(tmp_path / 'capture')
        capture_before = _digests(production.output_dir)
        with pytest.raises(ValueError, match='outside|target'):
            sim.evaluate(production, tmp_path / 'outside')
        assert _digests(production.output_dir) == capture_before
    assert _digests(sentinel) == before


def test_no_overwrite_and_independent_tally_results(make_model, synthetic,
                                                  unchanged, tmp_path):
    model, _ = make_model()
    model = deepcopy(model)
    model.tallies[0].higher_moments = True
    unchanged(model)
    config = d1s.Configuration(model)
    sim = _simulation([d1s.Irradiation(config, 100., 10.),
                        d1s.Cooling(config, 10., [0., 10.], name='observe')])
    result = sim.run(tmp_path / 'run')
    raw_before = _digests(tmp_path / 'run')
    for operation in (lambda: sim.run(tmp_path / 'run'),
                      lambda: sim.prepare(result.production.output_dir),
                      lambda: sim.evaluate(result.production,
                                           tmp_path / 'run' / 'observations')):
        with pytest.raises(FileExistsError):
            operation()
        assert _digests(tmp_path / 'run') == raw_before
    obs = result.observations['observe', 0.]
    physical = result.get_tally('all photons', 'observe', 0.)
    with openmc.StatePoint(obs.statepoint, autolink=False) as sp:
        raw = sp.tallies[physical.id]
        np.testing.assert_allclose(physical.mean, raw.mean * obs.source_rate)
        np.testing.assert_allclose(physical.std_dev, raw.std_dev * obs.source_rate)
        for attr, power in (('sum', 1), ('sum_sq', 2),
                            ('sum_third', 3), ('sum_fourth', 4)):
            np.testing.assert_allclose(getattr(physical, attr),
                                       getattr(raw, attr) * obs.source_rate**power)
    physical.mean[:] = -99.
    assert np.all(result.get_tally(physical.id, 1, (0., 's')).mean >= 0.)
    parent_tally = result.get_tally('by parent', 'observe', 0.)
    parent_filter = parent_tally.find_filter(openmc.ParentNuclideFilter)
    assert list(parent_filter.bins) == list(PARENTS)
    assert parent_tally.d1s_metadata['global_time'] == 100.
    assert parent_tally.d1s_metadata['normalization'] == 'physical rate'
    with pytest.raises(KeyError):
        result.get_tally('all photons', 'observe', 5.)
    with pytest.raises(KeyError):
        result.get_tally('missing', 'observe', 0.)
    assert _digests(tmp_path / 'run') == raw_before


@pytest.mark.parametrize('case', ['version', 'weight', 'parents', 'direction',
                                'particle', 'metadata', 'energy'])
def test_capture_schema_rejection(make_model, synthetic, monkeypatch, tmp_path,
                                  case):
    model, _ = make_model()
    sim = _simulation([d1s.Irradiation(d1s.Configuration(model), 1., 1.)])
    original = openmc.Model.run

    def corrupt(model, cwd, **kwargs):
        result = original(model, cwd, **kwargs)
        with h5py.File(Path(cwd) / 'd1s_source.0.h5', 'r+') as handle:
            if case == 'version':
                handle.attrs['d1s_version'] = 99
            elif case == 'weight':
                handle['source_histories'][()] = 0
            elif case == 'parents':
                handle['parent_nuclide'][0] = 999
            elif case == 'metadata':
                handle['cell_instance'][0] = -1
            else:
                bank = handle['source_bank'][()]
                if case == 'direction':
                    bank['u']['x'][0] = 2.
                elif case == 'particle':
                    bank['particle'][0] = int(openmc.ParticleType.NEUTRON)
                else:
                    bank['E'][0] = -1.
                handle['source_bank'][...] = bank
        return result

    monkeypatch.setattr(openmc.Model, 'run', corrupt)
    with pytest.raises(ValueError):
        sim.prepare(tmp_path / case)


def test_dagmc_raw_volume_scoping_auto_ids_and_instances(make_model, synthetic,
                                                       unchanged, monkeypatch,
                                                       tmp_path):
    # No DAGMC library is needed to read H5M tags or test selector membership.
    material = openmc.Material(name='test_mat')
    material.add_element('Ni', 1.)
    first = openmc.DAGMCUniverse(CAD, auto_geom_ids=True)
    second = openmc.DAGMCUniverse(CAD, auto_geom_ids=True)
    cells = []
    for i, universe in enumerate((first, first, second)):
        cell = openmc.Cell(fill=universe,
                           region=-openmc.Sphere(x0=20. * i, r=6.))
        cell.translation = (20. * i, 0., 0.)
        cells.append(cell)
    model = openmc.Model(geometry=openmc.Geometry(cells), materials=[material])
    model.settings.run_mode = 'fixed source'
    model.settings.batches = 3
    model.settings.particles = 10
    model.tallies = [_tally('CAD')]
    unchanged(model)
    assert _d1s._raw_volumes(CAD) == (1,)
    selectors = {'first-0': d1s.DAGMCVolumes(first, [1], [0]),
                 'first-1': d1s.DAGMCVolumes(first, [1], [1]),
                 'second': d1s.DAGMCVolumes(second, [1])}
    config = d1s.Configuration(model, selectors)
    sim = _simulation([d1s.Irradiation(config, 100., 1.)])
    assert sim.validate() is None
    plans, _ = synthetic
    bank = _sites(((0., 0., 0.), (20., 0., 0.), (40., 0., 0.)),
                  weights=(1., 2., 9.), energies=(1.e5, 2.e5, 3.e5))
    plans.append([{'bank': bank, 'cell_id': [70001, 70001, 80001],
                   'universe_id': [first.id, first.id, second.id],
                   'dagmc_volume_id': [1, 1, 1], 'cell_instance': [0, 1, 0]}])
    production = sim.prepare(tmp_path / 'cad-capture')
    np.testing.assert_array_equal(production.kernels[config].components,
                                  ['first-0', 'first-1', 'second'])
    # Exercise composition as well, without pretending to validate CAD points.
    preflights = []
    monkeypatch.setattr(_d1s, '_preflight',
                        lambda model, directory, bank: preflights.append(bank))
    evaluation = _simulation([sim.steps[0], d1s.Cooling(config, 1., [0.])])
    result = evaluation.evaluate(production, tmp_path / 'cad-evaluation')
    assert preflights == [result.observations['cooling-1', 0.].source_bank]
    for invalid in (d1s.DAGMCVolumes(first, [999]),
                    d1s.DAGMCVolumes(first, [1], [2]),
                    d1s.DAGMCVolumes(deepcopy(first), [1])):
        with pytest.raises(ValueError):
            _simulation([d1s.Cooling(d1s.Configuration(
                model, {'part': invalid}), 1.)]).validate()
    with pytest.raises(ValueError, match='Overlapping components'):
        _simulation([d1s.Cooling(d1s.Configuration(model, {
            'all': d1s.DAGMCVolumes(first, [1]),
            'one': d1s.DAGMCVolumes(first, [1], [1])}), 1.)]).validate()


@pytest.fixture
def transport_options():
    executable = shutil.which('openmc')
    if executable is None:
        pytest.skip('OpenMC executable is unavailable')
    xs = openmc.config.get('cross_sections')
    if xs is None or not Path(xs).is_file():
        pytest.skip('Continuous-energy cross sections are unavailable')
    return {'openmc_exec': executable, 'threads': 2, 'output': False}


def _unit_replay(model, observation, directory, options):
    replay = deepcopy(model)
    replay.settings.source = openmc.FileSource(observation.source_bank, strength=1.)
    replay.settings.photon_transport = True
    replay.settings.use_decay_photons = False
    replay.settings.d1s_source_write = False
    replay.settings.uniform_source_sampling = False
    replay.settings.output = {'summary': False, 'tallies': False, 'path': '.'}
    with openmc.config.patch('chain_file', CHAIN):
        result = Path(replay.run(cwd=directory, **options))
    return result if result.is_absolute() else directory / result


@pytest.fixture
def csg_transport(make_model, transport_options, tmp_path):
    first, a = make_model(strength=1.)
    second, b = make_model(strength=7.)
    target, c = make_model(radius=12.)
    configs = [d1s.Configuration(m, {'nickel': d1s.Cells([cell])})
               for m, cell in ((first, a), (second, b), (target, c))]
    def schedule(multiplier):
        return _simulation([
            d1s.Irradiation(configs[0], 3600., multiplier * 17.),
            d1s.Irradiation(configs[1], 7200., multiplier * 31.),
            d1s.Cooling(configs[2], 60., [0.], name='shutdown')])
    sim = schedule(1.)
    production = sim.prepare(tmp_path / 'capture', transport_options)
    result = sim.evaluate(production, tmp_path / 'evaluate', transport_options)
    scaled = schedule(3.).evaluate(production, tmp_path / 'scaled', transport_options)
    observation = result.observations['shutdown', 0.]
    reference = _unit_replay(target, observation, tmp_path / 'unit', transport_options)
    return configs, production, result, scaled, reference


def test_csg_capture_strength_invariance_and_analytic_rate(csg_transport):
    configs, production, result, _, _ = csg_transport
    a, b = (production.kernels[c] for c in configs[:2])
    assert len(a.bank) > 0
    assert a.source_weight == b.source_weight == 3000.
    np.testing.assert_array_equal(a.bank, b.bank)
    np.testing.assert_array_equal(a.parent_nuclide, b.parent_nuclide)
    obs = result.observations['shutdown', 0.]
    expected = 0.
    for kernel, duration, rate, age in ((a, 3600., 17., 7200.),
                                       (b, 7200., 31., 0.)):
        factors = np.array([_factor(p, duration, rate, age)
                            for p in kernel.parent_names])
        expected += np.sum(kernel.bank['wgt'] / kernel.source_weight
                           * factors[kernel.parent_nuclide])
        assert np.all(kernel.bank['particle'] == int(openmc.ParticleType.PHOTON))
        assert np.all(kernel.metadata['dagmc_volume_id'] == 0)
        assert np.all((kernel.metadata['batch'] >= 1)
                      & (kernel.metadata['batch'] <= 3))
    assert obs.source_rate == pytest.approx(expected)


def test_csg_parent_filter_and_source_metadata(csg_transport):
    _, _, result, _, _ = csg_transport
    obs = result.observations['shutdown', 0.]
    total = result.get_tally('all photons', 'shutdown', 0.)
    parent = result.get_tally('by parent', 'shutdown', 0.)
    assert not total.contains_filter(openmc.ParentNuclideFilter)
    assert list(parent.find_filter(openmc.ParentNuclideFilter).bins) == list(PARENTS)
    assert np.all(parent.mean > 0.)
    assert parent.mean.sum() <= total.mean.sum() * (1. + 1.e-12)
    assert total.mean.sum() > 0.
    assert parent.d1s_metadata['source_rate'] == obs.source_rate
    with h5py.File(obs.source_bank) as handle:
        weights = handle['sampling_weights'][()]
        assert np.ptp(weights) > 0.
        assert weights.sum() == pytest.approx(obs.source_rate)


def test_csg_physical_tallies_have_one_source_rate_factor(csg_transport):
    _, _, result, _, reference = csg_transport
    obs = result.observations['shutdown', 0.]
    with openmc.StatePoint(reference, autolink=False) as sp:
        for name in ('all photons', 'by parent'):
            tally = result.get_tally(name, 'shutdown', 0.)
            unit = sp.get_tally(name=name)
            np.testing.assert_allclose(tally.mean, unit.mean * obs.source_rate,
                                       rtol=2.e-12, atol=0.)
            np.testing.assert_allclose(tally.std_dev,
                                       unit.std_dev * obs.source_rate, rtol=1.e-10)


def test_csg_rate_scaling_is_linear_not_quadratic(csg_transport):
    _, _, result, scaled, _ = csg_transport
    assert scaled.source_rates['shutdown', 0.] == pytest.approx(
        3. * result.source_rates['shutdown', 0.])
    for name in ('all photons', 'by parent'):
        baseline = result.get_tally(name, 'shutdown', 0.)
        tripled = scaled.get_tally(name, 'shutdown', 0.)
        np.testing.assert_allclose(tripled.mean, 3. * baseline.mean, rtol=2.e-12)


def test_weighted_replay_samples_nonuniform_parent_probabilities(
        make_model, unchanged, transport_options, monkeypatch, tmp_path):
    """Check FileSource probabilities independently of the rate normalization.

    Equal-energy photons leave the same point in the same direction in an
    optically thin sphere, so parent flux fractions are binomial source counts.
    A uniform bank sampler would produce 1:1 rather than the required 1:9.
    """
    model, _ = make_model(radius=1.)
    model = deepcopy(model)
    model.materials[0].set_density('g/cm3', 1.e-20)
    unchanged(model)
    config = d1s.Configuration(model)
    sim = _simulation([d1s.Irradiation(config, 1.e12, 2.),
                        d1s.Cooling(config, 1., [0.])])

    def capture(model, directory, options, chain_file, capture):
        assert capture
        _write_capture(directory / 'd1s_source.0.h5', model, {
            'bank': _sites(((0., 0., 0.), (0., 0., 0.)),
                           weights=(1., 9.), energies=(1.e6, 1.e6))})
        return None

    with monkeypatch.context() as patch:
        patch.setattr(_d1s, '_transport', capture)
        production = sim.prepare(tmp_path / 'synthetic-capture')
    result = sim.evaluate(production, tmp_path / 'transport', transport_options)
    tally = result.get_tally('by parent', 'cooling-1', 0.)
    fractions = tally.mean.ravel() / tally.mean.sum()
    expected = np.array([.1, .9])
    sigma = np.sqrt(expected * (1. - expected) / 3000.)
    np.testing.assert_array_less(np.abs(fractions - expected), 6. * sigma)


@pytest.mark.skipif(not openmc.lib.feature_enabled('dagmc'),
                    reason='DAGMC CAD geometry is not enabled')
def test_dagmc_capture_transfer_and_weighted_replay(unchanged, transport_options,
                                                  tmp_path):
    def make(center):
        universe = openmc.DAGMCUniverse(CAD.resolve(), auto_geom_ids=True)
        # The fixture is tagged test_mat; Co59 is absent from chain_ni.xml,
        # so map its CAD material name to natural Ni activation instead.
        material = openmc.Material(name='test_mat')
        material.add_element('Ni', 1.)
        material.set_density('g/cm3', 8.9)
        boundary = openmc.Sphere(x0=center[0], r=6., boundary_type='vacuum')
        placement = openmc.Cell(fill=universe, region=-boundary)
        placement.translation = center
        placement.rotation = (0., 0., 90.)
        model = openmc.Model(geometry=openmc.Geometry([placement]),
                             materials=[material])
        model.settings.run_mode = 'fixed source'
        model.settings.batches = 3
        model.settings.particles = 1000
        model.settings.verbosity = 1
        model.settings.source = openmc.IndependentSource(
            space=openmc.stats.Point(center),
            energy=openmc.stats.Discrete([14.e6], [1.]))
        model.settings.output = {'summary': False, 'tallies': False}
        model.tallies = [_tally('all photons'), _tally('by parent', True)]
        unchanged(model)
        return d1s.Configuration(model, {'nickel': d1s.DAGMCVolumes(universe, [1])},
                                  {'nickel': _rigid(90., center)}), universe

    origin, origin_universe = make((0., 0., 0.))
    target, _ = make((20., 0., 0.))
    result = _simulation([d1s.Irradiation(origin, 3600., 17.),
        d1s.Cooling(target, 10., [0.])]).run(tmp_path / 'dagmc', transport_options)
    kernel = result.production.kernels[origin]
    assert len(kernel.bank) > 0
    assert kernel.source_weight == 3000.
    assert np.ptp(kernel.bank['wgt']) > 0.
    assert set(kernel.metadata['dagmc_volume_id']) == {1}
    assert set(kernel.metadata['universe_id']) == {origin_universe.id}
    assert np.all(kernel.metadata['cell_id'] != 1)
    obs = result.observations['cooling-1', 0.]
    assert obs.provenance['target_validation'] == 'isolated C API'
    with h5py.File(obs.source_bank) as handle:
        assert np.ptp(handle['sampling_weights'][()]) > 0.
        np.testing.assert_allclose(_vectors(handle['source_bank'][()], 'r'),
                                   _vectors(kernel.bank, 'r') + [20., 0., 0.])
        np.testing.assert_allclose(_vectors(handle['source_bank'][()], 'u'),
                                   _vectors(kernel.bank, 'u'))
    tally = result.get_tally('by parent', 'cooling-1', 0.)
    assert list(tally.find_filter(openmc.ParentNuclideFilter).bins) == list(PARENTS)
    assert np.all(tally.mean > 0.)
    reference = _unit_replay(target.model, obs, tmp_path / 'unit', transport_options)
    with openmc.StatePoint(reference, autolink=False) as sp:
        np.testing.assert_allclose(tally.mean,
            sp.get_tally(name='by parent').mean * obs.source_rate, rtol=2.e-12)