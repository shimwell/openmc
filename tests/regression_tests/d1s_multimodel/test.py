"""Analytic integration regression for multi-configuration D1S transport.

Real neutron captures and photon replays are checked against decay, source
normalization, rigid-transfer, and ancestry-partition identities. These physics
invariants are independent of the nuclear data library: no NNDC golden files
are generated or compared, including with --update. Suitable continuous-energy
nickel neutron and photon data must already be installed. Nothing is downloaded.
"""

from collections.abc import Mapping
from hashlib import sha256
from math import fsum
from pathlib import Path
import shutil

import h5py
import lxml.etree as ET
import numpy as np
import pytest

import openmc
import openmc.lib
from openmc.deplete import d1s
from tests.regression_tests import config


TESTS = Path(__file__).resolve().parents[2]
CHAIN = TESTS / 'chain_ni.xml'
CAD = TESTS / 'unit_tests' / 'dagmc' / 'dagmc_sphere_r5.h5m'
DURATIONS = (3600., 7200.)
RATES = (17., 31.)
COOLING_GAP = 600.
LATER_TIME = 1800.
OBSERVATIONS = (('shutdown', 0., 0.),
                ('later', LATER_TIME, COOLING_GAP + LATER_TIME))
BACKENDS = [
    pytest.param('csg', 'csg', id='CSG-to-CSG'),
    *[pytest.param(origin, target, id=f'{origin.upper()}-to-{target.upper()}',
                  marks=pytest.mark.skipif(
                      not openmc.lib.feature_enabled('dagmc'),
                      reason='DAGMC geometry is not enabled'))
      for origin, target in (('csg', 'dagmc'), ('dagmc', 'csg'),
                             ('dagmc', 'dagmc'))],
]


@pytest.fixture
def run_kwargs():
    """Use the regression suite's executable, MPI, and tracking options."""
    if shutil.which(config['exe']) is None:
        pytest.skip('Configured OpenMC executable is unavailable')
    xs = openmc.config.get('cross_sections')
    if xs is None or not Path(xs).is_file():
        pytest.skip('Continuous-energy cross sections are unavailable')
    library = openmc.data.DataLibrary.from_xml(xs)
    for name, kind in [(f'Ni{mass}', 'neutron')
                       for mass in (58, 60, 61, 62, 64)] + [('Ni', 'photon')]:
        entry = library.get_by_material(name, kind)
        if entry is None or not Path(entry['path']).is_file():
            pytest.skip(f'{kind} data for {name} are unavailable')
    options = {'openmc_exec': config['exe'], 'threads': 2, 'output': False,
               'event_based': config['event']}
    if config['mpi']:
        options['mpi_args'] = [config['mpiexec'], '-n', config['mpi_np']]
    return options


def _configuration(backend, energy, parents=(), moved=False):
    """Map one physical nickel component to unrelated model-local entities."""
    material = openmc.Material(name='test_mat')
    material.add_element('Ni', 1.)
    material.set_density('g/cm3', 8.9)
    if backend == 'dagmc':
        assert CAD.is_file()
        universe = openmc.DAGMCUniverse(CAD, auto_geom_ids=True)
        selector = d1s.DAGMCVolumes(universe, [1])
    else:
        # CSG births fit inside the faceted CAD sphere (nominal radius 5).
        # The larger CSG destination also accepts every CAD birth.
        surface = openmc.Sphere(r=6. if moved else 4.)
        nickel = openmc.Cell(fill=material, region=-surface)
        universe = openmc.Universe(cells=[
            nickel, openmc.Cell(region=+surface)])
        selector = d1s.Cells([nickel])

    transform = np.eye(4)
    if moved:
        transform[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        transform[:3, 3] = (20., -4., 3.)
    center = transform[:3, 3]
    boundary = openmc.Sphere(x0=center[0], y0=center[1], z0=center[2],
                             r=7., boundary_type='vacuum')
    placement = openmc.Cell(fill=universe, region=-boundary)
    if moved:
        placement.translation = center
        placement.rotation = (0., 0., 90.)
    model = openmc.Model(geometry=openmc.Geometry([placement]),
                         materials=[material])
    model.settings.run_mode = 'fixed source'
    model.settings.batches = 3
    model.settings.particles = 2000
    model.settings.seed = 1729
    model.settings.verbosity = 1
    model.settings.output = {'summary': False, 'tallies': False}
    model.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(center + [0.25, 0., 0.]),
        angle=openmc.stats.Monodirectional((1., 0., 0.)),
        energy=openmc.stats.Discrete([energy], [1.]),
        particle='photon' if moved else 'neutron')
    if parents:
        total = openmc.Tally(name='all photons')
        total.filters = [openmc.ParticleFilter('photon')]
        total.scores = ['flux']
        by_parent = openmc.Tally(name='by parent')
        by_parent.filters = [openmc.ParticleFilter('photon'),
                             openmc.ParentNuclideFilter(parents)]
        by_parent.scores = ['flux']
        model.tallies = [total, by_parent]
        # Descendant fluorescence and bremsstrahlung must retain ancestry too.
        model.settings.atomic_relaxation = True
        model.settings.electron_treatment = 'ttb'
    return d1s.Configuration(model, {'nickel': selector},
                              {'nickel': transform})


def _snapshot(value):
    """Freeze caller-owned state, including non-pickleable XML and caches."""
    seen = {}

    def freeze(obj):
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return obj
        if isinstance(obj, type):
            return ('type', obj.__module__, obj.__qualname__)
        if isinstance(obj, Path):
            return ('path', str(obj))
        if isinstance(obj, np.ndarray):
            return ('array', obj.dtype.descr, obj.shape, obj.tobytes())
        if isinstance(obj, ET._Element):
            return ('xml', ET.tostring(obj))
        if id(obj) in seen:
            return ('reference', seen[id(obj)])
        seen[id(obj)] = len(seen)
        if isinstance(obj, Mapping):
            return tuple((freeze(k), freeze(v)) for k, v in obj.items())
        if isinstance(obj, (tuple, list)):
            return (type(obj).__name__, tuple(map(freeze, obj)),
                    freeze(vars(obj)) if hasattr(obj, '__dict__') else None)
        if isinstance(obj, set):
            return ('set', tuple(sorted(map(repr, obj))))
        if hasattr(obj, '__dict__'):
            return (type(obj).__qualname__, freeze(vars(obj)))
        return (type(obj).__qualname__, repr(obj))

    return freeze(value)


def _digests(directory):
    return {p.relative_to(directory): sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def _irradiations(configurations, multiplier=1.):
    return [d1s.Irradiation(configuration, duration, multiplier * rate)
            for configuration, duration, rate in zip(
                configurations, DURATIONS, RATES)]


def _schedule(configurations, target, multiplier=1.):
    return d1s.Simulation([
        *_irradiations(configurations, multiplier),
        d1s.Cooling(target, COOLING_GAP, [0.], name='shutdown'),
        d1s.Cooling(target, LATER_TIME, [LATER_TIME], name='later'),
    ], chain_file=CHAIN)


def _vectors(bank, key):
    return np.column_stack([bank[key][axis] for axis in 'xyz'])


@pytest.mark.parametrize('neutron_backend,photon_backend', BACKENDS)
def test_d1s_multimodel(neutron_backend, photon_backend, run_kwargs,
                        run_in_tmpdir):
    """Two neutron spectra feed a distinct, translated/rotated photon model."""
    cwd = Path.cwd()
    global_config = dict(openmc.config)
    initialized = openmc.lib.is_initialized
    configurations = [_configuration(neutron_backend, energy)
                      for energy in (14.e6, 2.45e6)]
    snapshots = [_snapshot(c) for c in configurations]
    capture = d1s.Simulation(_irradiations(configurations), chain_file=CHAIN)
    production = capture.prepare(cwd / 'production', run_kwargs=run_kwargs)
    assert len(production.kernels) == 2
    assert set(production.kernels) == set(configurations)
    assert [_snapshot(c) for c in configurations] == snapshots

    ranks = int(config['mpi_np']) if config['mpi'] else 1
    parents = set()
    kernels = [production.kernels[c] for c in configurations]
    for configuration, kernel in zip(configurations, kernels):
        histories = (configuration.model.settings.batches
                     * configuration.model.settings.particles)
        assert len(kernel.files) == ranks
        shard_ranks = [int(p.name.split('.')[1]) for p in kernel.files]
        assert shard_ranks == list(range(ranks))
        counts, bank_sizes = [], []
        for filename in kernel.files:
            with h5py.File(filename) as handle:
                count = handle['source_histories']
                assert count.shape == () and count.dtype.kind == 'i'
                counts.append(int(count[()]))
                bank_sizes.append(len(handle['source_bank']))
        assert sum(counts) == kernel.source_histories == histories
        assert sum(bank_sizes) == len(kernel.bank) > 0
        assert kernel.statepoint.is_file()
        assert np.all(
            kernel.bank['particle'] == int(openmc.ParticleType.PHOTON))
        assert set(kernel.components) == {'nickel'}
        assert set(kernel.metadata['dagmc_volume_id']) == (
            {1} if neutron_backend == 'dagmc' else {0})
        # parent_names also contains unproduced chain entries. Select actual
        # birth tags, not a hand-picked subset such as just Co60 and Ni65.
        parents.update(np.asarray(kernel.parent_names)[kernel.parent_nuclide])
    assert not np.array_equal(kernels[0].bank, kernels[1].bank)
    parents = sorted(parents)
    assert parents

    target = _configuration(photon_backend, 1.e6, parents=parents, moved=True)
    target_snapshot = _snapshot(target)
    files_before = _digests(production.output_dir)
    baseline = _schedule(configurations, target).evaluate(
        production, cwd / 'baseline', run_kwargs=run_kwargs)
    doubled = _schedule(configurations, target, 2.).evaluate(
        production, cwd / 'doubled', run_kwargs=run_kwargs)
    assert baseline.production is doubled.production is production
    assert len(baseline.observations) == len(doubled.observations) == 2
    assert _digests(production.output_dir) == files_before

    decay = {parent: openmc.data.decay_constant(parent, chain_file=CHAIN)
             for parent in parents}
    assert all(lam > 0. for lam in decay.values())
    rotation = target.transforms['nickel'][:3, :3]
    translation = target.transforms['nickel'][:3, 3]
    shutdown_weights = None
    for step, local, cooling_age in OBSERVATIONS:
        observation = baseline.observations[step, local]
        scaled = doubled.observations[step, local]
        assert observation.global_time == sum(DURATIONS) + cooling_age
        assert observation.local_time == local
        expected_weights, positions, directions = [], [], []
        tags, kernel_ids = [], []
        for i, (kernel, duration, rate) in enumerate(
            zip(kernels, DURATIONS, RATES)):
            names = np.asarray(kernel.parent_names)[kernel.parent_nuclide]
            lam = np.array([decay[name] for name in names])
            # The first exposure ages throughout the SECOND exposure as well
            # as all preceding cooling intervals, not just this local offset.
            age = sum(DURATIONS[i + 1:]) + cooling_age
            weights = (kernel.bank['wgt'] / kernel.source_histories * rate
                       * -np.expm1(-lam * duration) * np.exp(-lam * age))
            assert weights.sum() > 0.
            expected_weights.append(weights)
            keep = weights > 0.
            positions.append(
                _vectors(kernel.bank[keep], 'r') @ rotation.T + translation)
            directions.append(_vectors(kernel.bank[keep], 'u') @ rotation.T)
            tags.extend(names[keep])
            kernel_ids.extend([i] * np.count_nonzero(keep))
        all_weights = np.concatenate(expected_weights)
        expected_rate = fsum(all_weights)
        assert observation.source_rate == pytest.approx(
            expected_rate, rel=2.e-12)
        assert scaled.source_rate == 2. * observation.source_rate
        if shutdown_weights is None:
            shutdown_weights = all_weights
        else:
            birth_tags = np.concatenate([
                np.asarray(k.parent_names)[k.parent_nuclide] for k in kernels])
            attenuation = np.exp(
                -np.array([decay[p] for p in birth_tags]) * cooling_age)
            np.testing.assert_allclose(
                all_weights, shutdown_weights * attenuation,
                rtol=2.e-12, atol=0.)
            assert (0. < observation.source_rate
                    < baseline.source_rates['shutdown', 0.])

        with (h5py.File(observation.source_bank) as handle,
              h5py.File(scaled.source_bank) as twice):
            replay = handle['source_bank'][()]
            weights = handle['sampling_weights'][()]
            np.testing.assert_allclose(weights, all_weights[all_weights > 0.],
                                       rtol=2.e-12, atol=0.)
            np.testing.assert_array_equal(
                twice['sampling_weights'][()], 2. * weights)
            np.testing.assert_array_equal(twice['source_bank'][()], replay)
            np.testing.assert_array_equal(
                handle['kernel_index'][()], kernel_ids)
            actual_tags = handle['parent_names'].asstr()[()][
                handle['parent_nuclide'][()]]
            np.testing.assert_array_equal(actual_tags, tags)
            np.testing.assert_allclose(
                _vectors(replay, 'r'), np.concatenate(positions),
                rtol=0., atol=1.e-12)
            np.testing.assert_allclose(
                _vectors(replay, 'u'), np.concatenate(directions),
                rtol=0., atol=1.e-12)
            assert np.all(replay['wgt'] == 1.)
            assert np.all(replay['surf_id'] == 0)
            assert handle['photon_source_rate'][()] == observation.source_rate

        total = baseline.get_tally('all photons', step, local)
        parent = baseline.get_tally('by parent', step, local)
        assert not total.contains_filter(openmc.ParentNuclideFilter)
        assert list(parent.find_filter(
            openmc.ParentNuclideFilter).bins) == parents
        assert np.all(np.isfinite(parent.mean)) and np.all(parent.mean >= 0.)
        assert parent.mean.sum() > 0.
        assert total.mean.sum() > 0.
        # The same photon histories, including descendants, are partitioned.
        # Rare/decayed parents can legitimately have zero sampled flux.
        assert parent.mean.sum() == pytest.approx(total.mean.sum(), rel=2.e-12)
        with (openmc.StatePoint(observation.statepoint, autolink=False) as sp,
              openmc.StatePoint(scaled.statepoint, autolink=False) as twice):
            assert sp.seed == twice.seed == target.model.settings.seed
            for tally in (total, parent):
                raw = sp.get_tally(name=tally.name)
                scaled_raw = twice.get_tally(name=tally.name)
                physical = doubled.get_tally(tally.name, step, local)
                # Exact single application of photons/s, not its square.
                np.testing.assert_array_equal(
                    tally.mean, raw.mean * observation.source_rate)
                np.testing.assert_array_equal(
                    tally.std_dev, raw.std_dev * observation.source_rate)
                # Identical replay seeds/probabilities: only FP reduction
                # roundoff is allowed, not an independent-history tolerance.
                np.testing.assert_allclose(
                    scaled_raw.mean, raw.mean, rtol=2.e-12, atol=0.)
                np.testing.assert_allclose(physical.mean, 2. * tally.mean,
                                           rtol=2.e-12, atol=0.)

    assert [_snapshot(c) for c in configurations] == snapshots
    assert _snapshot(target) == target_snapshot
    assert dict(openmc.config) == global_config
    assert openmc.lib.is_initialized == initialized
    assert Path.cwd() == cwd