"""Multi-configuration D1S photon-birth capture and replay.

D1S weights reaction-produced decay photons; it does not evolve compositions
or calculate a depletion inventory. The capture and weighted FileSource replay
extensions are required in the OpenMC executable used for nonzero runs.
"""

from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import json
from math import fsum
from numbers import Integral, Real
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import h5py
import numpy as np

import openmc
import openmc.checkvalue as cv
from openmc.statepoint import _VERSION_STATEPOINT


__all__ = [
    'Cells', 'DAGMCVolumes', 'Configuration', 'Irradiation', 'Cooling',
    'Simulation', 'Production', 'Observation', 'Results',
]

_TIME_UNITS = {
    's': 1., 'sec': 1., 'min': 60., 'minute': 60.,
    'h': 3600., 'hr': 3600., 'hour': 3600., 'd': 86400., 'day': 86400.,
    'a': 31557600., 'year': 31557600.,
}
_POSITION_DTYPE = np.dtype([(axis, '<f8') for axis in 'xyz'])
_SITE_DTYPE = np.dtype([
    ('r', _POSITION_DTYPE), ('u', _POSITION_DTYPE), ('E', '<f8'),
    ('time', '<f8'), ('wgt', '<f8'), ('delayed_group', '<i4'),
    ('surf_id', '<i4'), ('particle', '<i4'),
])
_METADATA = (
    'cell_id', 'cell_instance', 'universe_id', 'dagmc_volume_id',
    'history_id', 'batch',
)
_UNCERTAINTY = (
    'Standard deviations include photon transport uncertainty conditional on '
    'the sampled neutron birth banks, not neutron bank generation uncertainty.'
)


def _real(name, value, positive=False):
    cv.check_type(name, value, Real)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be a real number, not bool')
    value = float(value)
    if not np.isfinite(value) or value < 0. or (positive and value == 0.):
        qualifier = 'positive' if positive else 'nonnegative'
        raise ValueError(f'{name} must be finite and {qualifier}')
    return value


def _seconds(value, name='time', positive=False):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f'{name} must be seconds or a (value, unit) pair')
        value, unit = value
        cv.check_type(f'{name} unit', unit, str)
        cv.check_value(f'{name} unit', unit, _TIME_UNITS)
        value = _real(name, value, positive) * _TIME_UNITS[unit]
    return _real(name, value, positive)


def _integer(name, value, minimum=0):
    cv.check_type(name, value, Integral)
    if isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _name(name, value):
    cv.check_type(name, value, str)
    if not value.strip():
        raise ValueError(f'{name} must not be empty')
    return value


def _pairs(values, instances, name, object_type=None):
    cv.check_type(name, values, Iterable)
    values = tuple(values)
    if not values:
        raise ValueError(f'{name} must not be empty')
    if instances is None:
        instances = (None,) * len(values)
    else:
        cv.check_type('instances', instances, Iterable)
        instances = tuple(instances)
        if len(instances) != len(values):
            raise ValueError(
                'instances must have one entry per selected entity')
    pairs = []
    for value, instance in zip(values, instances):
        if object_type is not None and isinstance(value, object_type):
            uid = value.id
        else:
            uid = _integer(name, value, 1)
        if instance is not None:
            instance = _integer('instance', instance)
        pairs.append((uid, instance))
    for i, (uid, instance) in enumerate(pairs):
        if any(
            uid == other
            and (instance is None or inst is None or inst == instance)
            for other, inst in pairs[:i]
        ):
            raise ValueError(f'Duplicate or overlapping {name} selections')
    return values, tuple(pairs)


class Cells:
    """Select CSG leaf material cells, optionally by occurrence.

    Parameters
    ----------
    cells : iterable of openmc.Cell or int
        Leaf material/distributed-material cells or positive model-local IDs.
        Containers, void cells, and DAGMCCells are not accepted. Cell objects
        must be the actual objects in the configuration's model.
    instances : iterable of int or None, optional
        Parallel to ``cells``: zero-based cell instance numbers, with None
        selecting every occurrence of that cell. Omitting this argument selects
        every occurrence. To select multiple occurrences, repeat a cell with
        different instance entries, e.g. ``Cells([cell, cell], [0, 2])``.
        Capture membership always uses each original site's instance, not a
        renumbered/target instance. A target selector independently specifies
        the allowed destination instances.
    """

    def __init__(self, cells, instances=None):
        self.cells, self._pairs = _pairs(
            cells, instances, 'cells', openmc.Cell)
        self.instances = tuple(instance for _, instance in self._pairs)
        for cell in self.cells:
            if isinstance(cell, openmc.Cell):
                self._check_leaf(cell)

    @staticmethod
    def _check_leaf(cell):
        if (isinstance(cell, openmc.DAGMCCell)
                or cell.fill_type not in ('material', 'distribmat')):
            raise ValueError(
                'Cells requires CSG leaf material cells; use '
                'DAGMCVolumes for CAD volumes')

    def _mask(self, metadata):
        mask = np.zeros(len(metadata['cell_id']), dtype=bool)
        for uid, instance in self._pairs:
            selected = metadata['cell_id'] == uid
            if instance is not None:
                selected &= metadata['cell_instance'] == instance
            mask |= selected
        return mask & (metadata['dagmc_volume_id'] == 0)


class DAGMCVolumes:
    """Select raw H5M volume IDs scoped to a DAGMC universe.

    Parameters
    ----------
    universe : openmc.DAGMCUniverse
        Actual universe object in the configuration's model. Its ID scopes
        raw CAD volume IDs; OpenMC's runtime/auto_geom_ids cell IDs are never
        used as physical identity.
    volume_ids : iterable of int
        Positive raw H5M GLOBAL_ID values for three-dimensional entities.
    instances : iterable of int or None, optional
        Parallel to ``volume_ids``, as for :class:`Cells`. These are zero-based
        occurrences of a volume in repeated placements of the universe.

    Notes
    -----
    Raw IDs are checked directly against H5M tags. Destination points are
    checked in an isolated C API subprocess before transport, including CAD
    volume membership and instances. This requires a DAGMC-capable Python
    shared library as well as a DAGMC-capable transport executable.
    """

    def __init__(self, universe, volume_ids, instances=None):
        cv.check_type('universe', universe, openmc.DAGMCUniverse)
        self.universe = universe
        self.volume_ids, self._pairs = _pairs(
            volume_ids, instances, 'volume IDs')
        self.instances = tuple(instance for _, instance in self._pairs)

    def _mask(self, metadata):
        mask = np.zeros(len(metadata['cell_id']), dtype=bool)
        for uid, instance in self._pairs:
            selected = metadata['dagmc_volume_id'] == uid
            if instance is not None:
                selected &= metadata['cell_instance'] == instance
            mask |= selected
        return mask & (metadata['universe_id'] == self.universe.id)


def _transform(value):
    if np.iscomplexobj(value):
        raise ValueError('Rigid transforms must be real-valued')
    matrix = np.array(value, dtype=float, copy=True)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Transforms must be finite 4x4 rigid matrices')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], rtol=0., atol=1.e-12)
            or not np.allclose(
                rotation.T @ rotation, np.eye(3), rtol=0., atol=1.e-10)
            or not np.isclose(np.linalg.det(rotation), 1.,
                              rtol=0., atol=1.e-10)):
        raise ValueError(
            'Transforms require a proper rotation, without scaling '
            'or reflection, and homogeneous last row [0, 0, 0, 1]')
    matrix.setflags(write=False)
    return matrix


class Configuration:
    """A transport model and stable physical component identities.

    Parameters
    ----------
    model : openmc.Model
        Caller-owned model, never modified by this workflow. Use fixed-source,
        continuous-energy Monte Carlo settings. Do not change a configuration
        after preparing production; create another Configuration whenever the
        neutron model or source shape changes. Source rates belong to steps.
    components : mapping of str to Cells or DAGMCVolumes or None, optional
        Stable physical identity to model-local selector. Every captured photon
        must match exactly one non-None selector. Explicit None removes a
        component from this configuration without discarding its history.
        Historical identities must be present in every subsequent target map;
        an omitted identity is an error, not implicit removal. Passive
        shielding need not appear in the map. With None, all sources retain
        their global coordinates; only other no-map configurations may be
        paired with it.
    transforms : mapping of str to array-like, optional
        Proper rigid 4x4 matrices mapping reference component coordinates to
        this model's global coordinates (cm). Omitted entries are identities.
        Transfer uses ``T_target @ inv(T_origin)`` on positions and its
        rotation on directions. No spatial or angular resampling is performed.
    """

    def __init__(self, model, components=None, transforms=None):
        cv.check_type('model', model, openmc.Model)
        self.model = model
        self._base_dir = Path.cwd().resolve()
        if components is not None:
            cv.check_type('components', components, Mapping)
            components = dict(components)
            for key, selector in components.items():
                _name('component identity', key)
                cv.check_type('component selector', selector,
                              (Cells, DAGMCVolumes), none_ok=True)
            components = MappingProxyType(components)
        self.components = components
        if transforms is None:
            transforms = {}
        cv.check_type('transforms', transforms, Mapping)
        if transforms and components is None:
            raise ValueError('Transforms require a component map')
        normalized = {}
        for key, matrix in transforms.items():
            if key not in components:
                raise ValueError(f'Transform for unknown component {key!r}')
            normalized[key] = _transform(matrix)
        self.transforms = MappingProxyType(normalized)

    def _matrix(self, component):
        return self.transforms.get(component, np.eye(4))


class Irradiation:
    """An irradiation interval (not a depletion step).

    Parameters
    ----------
    configuration : Configuration
        Neutron production state. Reusing the very same object reuses its bank.
    duration : float or (float, str)
        Positive finite duration, in seconds or with units s, min, h, d, a
        (Julian years); sec, minute, hr, hour, day, year are also accepted.
    source_rate : float
        Nonnegative finite initial neutron source rate in neutrons/s. Zero
        advances time without generating production.
    name : str, optional
        Unique schedule key; default keys are irradiation-1, irradiation-2, ...
    """

    def __init__(self, configuration, duration, source_rate, name=None):
        cv.check_type('configuration', configuration, Configuration)
        self.configuration = configuration
        self.duration = _seconds(duration, 'duration', positive=True)
        self.source_rate = _real('source_rate', source_rate)
        self.name = None if name is None else _name('step name', name)


class Cooling:
    """A cooling interval with local-time photon observations.

    Parameters
    ----------
    configuration : Configuration
        Model in which historical photon births will be replayed.
    duration : float or (float, str)
        Positive finite interval duration; units as for :class:`Irradiation`.
    sample_times : iterable of float or (float, str), optional
        Unique local offsets in [0, duration], in seconds or explicit units.
        Zero observes the post-configuration-switch state. The empty default
        requests no snapshots, not an implicit end-of-step observation.
    name : str, optional
        Unique schedule key; default keys are cooling-1, cooling-2, ...
    """

    def __init__(self, configuration, duration, sample_times=(), name=None):
        cv.check_type('configuration', configuration, Configuration)
        self.configuration = configuration
        self.duration = _seconds(duration, 'duration', positive=True)
        cv.check_type('sample_times', sample_times, Iterable)
        times = tuple(_seconds(t, 'sample time') for t in sample_times)
        if any(t > self.duration for t in times):
            raise ValueError('Cooling sample times must lie in [0, duration]')
        if len(set(times)) != len(times):
            raise ValueError('Cooling sample times must be unique')
        self.sample_times = tuple(sorted(times))
        self.name = None if name is None else _name('step name', name)


def _absolute(path, base):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _clone(configuration):
    """Deep-copy, retaining IDs, and resolve inputs before Model.run changes
    cwd.
    """
    model = deepcopy(configuration.model)
    base = configuration._base_dir
    seen = set()

    def visit(obj):
        if id(obj) in seen or isinstance(obj, (str, bytes, Real, Path)):
            return
        seen.add(id(obj))
        if isinstance(obj, (openmc.DAGMCUniverse, openmc.UnstructuredMesh)):
            obj.filename = _absolute(obj.filename, base)
        elif isinstance(obj, openmc.FileSource):
            obj.path = _absolute(obj.path, base)
        elif isinstance(obj, openmc.CompiledSource):
            obj.library = _absolute(obj.library, base)
        if isinstance(obj, Mapping):
            children = obj.values()
        elif isinstance(obj, np.ndarray):
            children = obj.flat if obj.dtype.kind == 'O' else ()
        elif isinstance(obj, (list, tuple)):
            children = obj
        elif (type(obj).__module__.startswith('openmc.')
              and hasattr(obj, '__dict__')):
            children = tuple(vars(obj).values())
        else:
            return
        for child in children:
            visit(child)

    visit(model)
    # XML exports cells in ascending ID order. Instance numbering must use
    # that same traversal, rather than the caller's insertion order.
    for universe in model.geometry.get_all_universes().values():
        if not isinstance(universe, openmc.DAGMCUniverse):
            universe._cells = dict(sorted(universe.cells.items()))
    xs = model.materials.cross_sections or openmc.config.get('cross_sections')
    if xs is not None:
        model.materials.cross_sections = _absolute(xs, base)
        if not model.materials:
            model.materials.extend(model.geometry.get_all_materials().values())
    for key in ('properties_file', 'weight_windows_file'):
        value = getattr(model.settings, key, None)
        if value is not None:
            setattr(model.settings, key, _absolute(value, base))
    if model.settings.surf_source_read:
        ssr = dict(model.settings.surf_source_read)
        if 'path' in ssr:
            ssr['path'] = _absolute(ssr['path'], base)
        model.settings.surf_source_read = ssr
    model.settings.output = dict(model.settings.output or {}, path='.')
    return model


def _raw_volumes(filename):
    """Read raw volume IDs in MOAB entity-handle order, never runtime IDs."""
    with h5py.File(filename, 'r') as handle:
        tags = handle['tstt/tags']
        dimensions = tags['GEOM_DIMENSION']
        entities = dimensions['id_list'][()]
        entities = entities[dimensions['values'][()].reshape(-1) == 3]
        global_ids = tags['GLOBAL_ID']
        if 'values' in global_ids:
            ids = dict(zip(global_ids['id_list'][()],
                           global_ids['values'][()].reshape(-1)))
            volumes = [(int(entity), int(ids[entity])) for entity in entities]
        else:
            start = int(handle['tstt/sets/list'].attrs['start_id'])
            ids = handle['tstt/sets/tags/GLOBAL_ID'][()].reshape(-1)
            volumes = [(int(entity), int(ids[int(entity) - start]))
                       for entity in entities]
    volumes.sort()
    result = tuple(uid for _, uid in volumes)
    if len(set(result)) != len(result) or any(uid <= 0 for uid in result):
        raise ValueError(f'Invalid raw DAGMC volume IDs in {filename}')
    return result


def _universe_counts(geometry):
    counts = {}

    def visit(universe):
        counts[universe.id] = counts.get(universe.id, 0) + 1
        if isinstance(universe, openmc.DAGMCUniverse):
            return
        for cell in universe.cells.values():
            if cell.fill_type == 'universe':
                visit(cell.fill)
            elif cell.fill_type == 'lattice':
                for index in cell.fill._natural_indices:
                    visit(cell.fill.get_universe(index))

    visit(geometry.root_universe)
    return counts


def _validate_configuration(configuration):
    original = configuration.model
    settings = original.settings
    if (settings.run_mode != 'fixed source'
            or settings.energy_mode not in (None, 'continuous-energy')
            or settings.random_ray):
        raise ValueError(
            'D1S requires fixed-source continuous-energy Monte Carlo')
    if original.geometry.root_universe is None:
        raise ValueError('A D1S configuration requires a geometry')
    model = _clone(configuration)
    geometry = model.geometry
    geometry.determine_paths()
    cells = geometry.get_all_cells()
    original_cells = original.geometry.get_all_cells()
    universes = original.geometry.get_all_universes()
    universe_counts = _universe_counts(geometry)
    selections = []
    for component, selector in (configuration.components or {}).items():
        if selector is None:
            continue
        if isinstance(selector, Cells):
            for original_cell, (uid, instance) in zip(
                selector.cells, selector._pairs
            ):
                if uid not in cells:
                    raise ValueError(
                        f'Component {component!r}: cell {uid} not in model')
                if (isinstance(original_cell, openmc.Cell)
                        and original_cells[uid] is not original_cell):
                    raise ValueError(
                        f'Component {component!r}: cell object not in model')
                selector._check_leaf(cells[uid])
                count = cells[uid].num_instances
                if instance is not None and instance >= count:
                    raise ValueError(f'Cell {uid} has no instance {instance}')
                selections.append((component, ('csg', uid), instance))
        else:
            universe = selector.universe
            if universes.get(universe.id) is not universe:
                raise ValueError(
                    f'Component {component!r}: DAGMC universe not in model')
            volumes = _raw_volumes(
                _absolute(universe.filename, configuration._base_dir))
            for uid, instance in selector._pairs:
                if uid not in volumes:
                    raise ValueError(
                        f'Raw volume {uid} not in DAGMC universe '
                        f'{universe.id}')
                if (instance is not None
                        and instance >= universe_counts[universe.id]):
                    raise ValueError(
                        f'DAGMC volume {uid} has no instance {instance}')
                selections.append(
                    (component, ('dagmc', universe.id, uid), instance))
    for i, (component, entity, instance) in enumerate(selections):
        for other, other_entity, other_instance in selections[:i]:
            if (entity == other_entity
                    and (instance is None or other_instance is None
                         or instance == other_instance)):
                raise ValueError(
                    f'Overlapping components {other!r} and {component!r}')
    return model


def _uninitialized():
    # Model.is_initialized checks the global library, not model ownership.
    if openmc.Model().is_initialized:
        raise RuntimeError(
            'D1S cannot run while openmc.lib is initialized; '
            'finalize it explicitly first. Its state was not changed.')


def _run_options(run_kwargs):
    if run_kwargs is None:
        return {}
    cv.check_type('run_kwargs', run_kwargs, Mapping)
    options = deepcopy(dict(run_kwargs))
    allowed = {'particles', 'threads', 'geometry_debug', 'tracks', 'output',
               'openmc_exec', 'mpi_args', 'event_based'}
    unexpected = options.keys() - allowed
    if unexpected:
        raise ValueError(
            f'D1S controls input/output and restart settings; unsupported '
            f'run_kwargs: {sorted(unexpected)}')
    executable = options.get('openmc_exec')
    if (executable is not None
            and (isinstance(executable, Path) or '/' in str(executable))):
        options['openmc_exec'] = str(Path(executable).expanduser().resolve())
    return options


def _new_directory(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(
                f'D1S output directory must be new or empty: {path}')
    else:
        path.mkdir(parents=True)
    return path


def _capture_setting(settings, enabled):
    if not hasattr(type(settings), 'd1s_source_write'):
        raise RuntimeError(
            'This OpenMC installation lacks Settings.d1s_source_write '
            'and the D1S capture/weighted replay backend')
    settings.d1s_source_write = enabled


def _transport(model, directory, options, chain_file, capture):
    _uninitialized()
    _capture_setting(model.settings, capture)
    model.settings.photon_transport = True
    model.settings.use_decay_photons = capture
    model.settings.run_mode = 'fixed source'
    model.settings.energy_mode = 'continuous-energy'
    model.settings.statepoint = {'batches': [model.settings.batches]}
    with ExitStack() as stack:
        if chain_file is not None:
            stack.enter_context(openmc.config.patch('chain_file', chain_file))
        xs = model.materials.cross_sections
        if xs is not None:
            stack.enter_context(openmc.config.patch('cross_sections', xs))
        result = model.run(cwd=directory, **options)
    if result is not None:
        result = Path(result)
        result = (result.resolve() if result.is_absolute()
                  else (directory / result).resolve())
    return result


def _strings(dataset):
    if dataset.ndim != 1 or h5py.check_string_dtype(dataset.dtype) is None:
        raise ValueError('parent_names must be a one-dimensional string array')
    return tuple(dataset.asstr()[()].tolist())


def _integer_array(handle, key, size):
    values = handle[key][()]
    if values.shape != (size,) or values.dtype.kind not in 'iu':
        raise ValueError(
            f'{handle.filename}: {key} must be a length-{size} integer array')
    return values


@dataclass(frozen=True)
class _Kernel:
    configuration: Configuration
    bank: np.ndarray
    parent_names: tuple
    parent_nuclide: np.ndarray
    metadata: Mapping
    components: np.ndarray
    source_weight: float
    source_histories: int
    files: tuple
    statepoint: Path | None
    version: tuple


def _read_kernel(configuration, files, statepoint=None):
    banks, parents, totals, metadata, components = [], [], [], [], []
    history_counts = []
    names, name_index = [], {}
    versions = set()
    for filename in files:
        with h5py.File(filename, 'r') as handle:
            filetype = handle.attrs.get('filetype', '')
            if isinstance(filetype, bytes):
                filetype = filetype.decode()
            version = tuple(
                np.asarray(handle.attrs.get('version', ())).tolist())
            if (filetype != 'source' or len(version) != 2
                    or version[0] != _VERSION_STATEPOINT
                    or handle.attrs.get('d1s_version') != 1):
                raise ValueError(
                    f'{filename}: unsupported D1S source file format')
            versions.add(version)
            total = handle['source_weight'][()]
            if np.ndim(total) != 0:
                raise ValueError(f'{filename}: source_weight must be scalar')
            totals.append(_real('source_weight', total))
            count = handle['source_histories'][()]
            if np.ndim(count) != 0:
                raise ValueError(
                    f'{filename}: source_histories must be scalar')
            history_counts.append(_integer('source_histories', count))
            raw = handle['source_bank'][()]
            if raw.ndim != 1 or raw.dtype.names != _SITE_DTYPE.names:
                raise ValueError(
                    f'{filename}: invalid ordinary OpenMC source_bank dtype')
            bank = raw.astype(_SITE_DTYPE)
            for key in ('E', 'time', 'wgt'):
                if not np.isfinite(bank[key]).all() or (bank[key] < 0.).any():
                    raise ValueError(f'{filename}: invalid source {key}')
            for key in ('r', 'u'):
                vectors = _vectors(bank, key)
                if not np.isfinite(vectors).all():
                    raise ValueError(f'{filename}: nonfinite source {key}')
                if (key == 'u'
                        and not np.allclose(
                            np.linalg.norm(vectors, axis=1), 1.)):
                    raise ValueError(
                        f'{filename}: source directions must be unit vectors')
            if (bank['particle'] != int(openmc.ParticleType.PHOTON)).any():
                raise ValueError(
                    f'{filename}: capture bank contains non-photons')
            local_names = _strings(handle['parent_names'])
            if len(set(local_names)) != len(local_names):
                raise ValueError(f'{filename}: duplicate parent names')
            local = _integer_array(handle, 'parent_nuclide', len(bank))
            if ((local < 0).any() or (local >= len(local_names)).any()
                    or any(not name.strip() for name in local_names)):
                raise ValueError(
                    f'{filename}: invalid parent-nuclide metadata')
            indices = []
            for name in local_names:
                if name not in name_index:
                    name_index[name] = len(names)
                    names.append(name)
                indices.append(name_index[name])
            parents.append(np.asarray(indices, dtype=np.int64)[local])
            meta = {key: _integer_array(handle, key, len(bank))
                    for key in _METADATA}
            if any((values < 0).any() for values in meta.values()):
                raise ValueError(f'{filename}: negative D1S site metadata')
            assigned = np.full(len(bank), None, dtype=object)
            if configuration.components is not None:
                count = np.zeros(len(bank), dtype=int)
                for component, selector in configuration.components.items():
                    if selector is not None:
                        mask = selector._mask(meta)
                        count += mask
                        assigned[mask] = component
                if (count != 1).any():
                    raise ValueError(
                        f'{filename}: every captured photon must belong '
                        'to exactly one component (missing or overlapping '
                        'selection)')
            banks.append(bank)
            metadata.append(meta)
            components.append(assigned)
    if len(versions) != 1:
        raise ValueError(
            'D1S shards must have a common statepoint format version')
    total = fsum(totals)
    histories = sum(history_counts)
    if not np.isfinite(total) or histories <= 0:
        raise ValueError(
            'D1S requires finite source_weight and positive source_histories')
    arrays = {key: np.concatenate([m[key] for m in metadata])
              for key in _METADATA}
    bank = np.concatenate(banks)
    parent = np.concatenate(parents)
    component_array = np.concatenate(components)
    for array in (bank, parent, component_array, *arrays.values()):
        array.setflags(write=False)
    return _Kernel(
        configuration, bank, tuple(names), parent, MappingProxyType(arrays),
        component_array, total, histories, tuple(files), statepoint,
        versions.pop())


class Production:
    """Reusable D1S production from :meth:`openmc.deplete.d1s.Simulation.prepare`.

    ``kernels`` maps the original Configuration objects (by identity) to their
    immutable captured arrays, global initial-neutron ``source_histories``,
    informational ``source_weight``, shard ``files``, and raw ``statepoint``
    path. Rates and durations are deliberately
    absent. Reuse in another Simulation requires the same irradiation
    Configuration objects and decay-data file. This is an in-process object,
    not a depletion result or an on-disk restart format. Configuration objects
    and their models must not be modified after preparation.
    """

    def __init__(self, kernels, output_dir, chain_file):
        self.kernels = MappingProxyType(dict(kernels))
        self.output_dir = Path(output_dir)
        self.chain_file = chain_file


@dataclass(frozen=True)
class Observation:
    """One cooling snapshot, with all times in seconds.

    ``step`` is its unique schedule key; ``step_index`` is zero-based.
    ``source_rate`` is the total physical photons/s after component removal.
    ``statepoint`` is the unscaled backend output path, or None for zero
    source.
    ``source_bank`` is the replay bank, or None for zero source. ``tallies``
    maps tally IDs to independent, physically normalized in-memory tallies.
    ``provenance`` records contributing kernels and validation/uncertainty
    notes.
    """

    step: str
    step_index: int
    local_time: float
    global_time: float
    source_rate: float
    statepoint: Path | None
    source_bank: Path | None
    tallies: Mapping
    provenance: Mapping


class Results:
    """Physical-rate tallies and snapshot metadata, not depletion inventories.

    ``observations[(step_name, local_seconds)]`` is an :class:`Observation`.
    ``source_rates`` and ``statepoint_paths`` use the same keys; zero-source
    observations have rate zero and path None. ``provenance`` explains the
    normalization and conditional uncertainty. Raw statepoints are unchanged.
    Tally score/filter units are retained with the per-source normalization
    replaced by photons/s (e.g. flux becomes particle-cm/s, before volume
    normalization). No ParentNuclideFilter is inserted, removed, or summed.
    """

    def __init__(self, observations, step_names, production):
        self.observations = MappingProxyType(dict(observations))
        self.step_names = tuple(step_names)
        self.production = production
        self.source_rates = MappingProxyType({
            k: v.source_rate for k, v in self.observations.items()})
        self.statepoint_paths = MappingProxyType({
            k: v.statepoint for k, v in self.observations.items()})
        self.provenance = MappingProxyType({
            'method': 'D1S photon-birth capture/replay; no depletion',
            'normalization': (
                'raw birth weight / global initial neutron histories '
                '* interval decay factor, then photon tally * photons/s'),
            'uncertainty': _UNCERTAINTY,
        })

    def get_tally(self, name, step, time):
        """Return an isolated physical-rate Tally at a cooling observation.

        Parameters
        ----------
        name : str or int
            Unique tally name or tally ID in the observation's target model.
        step : str or int
            Unique step name or zero-based schedule index.
        time : float or (float, str)
            Local cooling offset, in seconds or explicit units. It must match a
            requested sample time (within roundoff, not a nearest-time lookup).

        Returns
        -------
        openmc.Tally
            Independent copy with ``d1s_metadata`` giving step, local/global
            seconds, photon source rate and the conditional uncertainty note.
        """
        if isinstance(step, Integral) and not isinstance(step, bool):
            step = self.step_names[_integer('step index', step)]
        else:
            cv.check_type('step', step, str)
        seconds = _seconds(time)
        matches = [
            obs for (key, local), obs in self.observations.items()
            if key == step
            and np.isclose(local, seconds, rtol=1.e-12, atol=1.e-12)]
        if len(matches) != 1:
            raise KeyError(
                f'No unique observation for step {step!r}, '
                f'local time {seconds} s')
        observation = matches[0]
        if isinstance(name, Integral):
            tallies = [t for uid, t in observation.tallies.items()
                       if uid == name]
        else:
            cv.check_type('tally name', name, str)
            tallies = [t for t in observation.tallies.values()
                       if t.name == name]
        if len(tallies) != 1:
            raise KeyError(
                f'No unique tally {name!r} at {step!r}, {seconds} s')
        return deepcopy(tallies[0])


def _vectors(bank, key):
    return np.column_stack([bank[key][axis] for axis in 'xyz'])


def _set_vectors(bank, key, values):
    for i, axis in enumerate('xyz'):
        bank[key][axis] = values[:, i]


def _csg_points(model, selector, bank):
    geometry = model.geometry
    geometry.determine_paths()
    for point in _vectors(bank, 'r'):
        # Universe.find applies placements in-place, so never pass a bank view.
        traversed = geometry.find(point.copy())
        leaf = traversed[-1] if traversed else None
        if not isinstance(leaf, openmc.Cell):
            raise ValueError(
                f'Transferred D1S point {point} is outside '
                'target material cells')
        path = []
        for entity in traversed:
            if isinstance(entity, tuple):
                lattice, indices = entity
                path.append(
                    f'l{lattice.id}({",".join(str(x) for x in indices)})')
            elif isinstance(entity, openmc.Cell):
                path.append(f'c{entity.id}')
            elif isinstance(entity, openmc.UniverseBase):
                path.append(f'u{entity.id}')
        instance = geometry.get_instances('->'.join(path))
        if selector is None:
            continue
        if not any(uid == leaf.id and (inst is None or inst == instance)
                   for uid, inst in selector._pairs):
            raise ValueError(
                f'Transferred D1S point {point} does not lie in its '
                'selected target cell/instance')


def _compose(target, model, contributions, filename):
    banks, weights, parents, component_ids, kernel_ids = [], [], [], [], []
    metadata = {key: [] for key in _METADATA}
    names, name_indices, used, specs = [], {}, [], []
    component_names = list(target.components or {})
    has_dagmc = any(
        isinstance(u, openmc.DAGMCUniverse)
        for u in model.geometry.get_all_universes().values())
    for kernel_index, (kernel, factors) in enumerate(contributions):
        physical = (kernel.bank['wgt'] / kernel.source_histories
                    * factors[kernel.parent_nuclide])
        if not np.isfinite(physical).all() or (physical < 0.).any():
            raise ValueError('Nonfinite or negative D1S emission weights')
        keep = physical > 0.
        bank = kernel.bank.copy()
        target_components = np.full(len(bank), -1, dtype=np.int64)
        if target.components is not None:
            for component, origin_selector in (
                kernel.configuration.components.items()
            ):
                if origin_selector is None:
                    continue
                if component not in target.components:
                    raise ValueError(
                        f'Target is missing historical component '
                        f'{component!r}')
                selected = kernel.components == component
                selector = target.components[component]
                if selector is None:
                    keep[selected] = False
                    continue
                target_components[selected] = component_names.index(component)
                transfer = target._matrix(component) @ np.linalg.inv(
                    kernel.configuration._matrix(component))
                points = (_vectors(bank[selected], 'r') @ transfer[:3, :3].T
                          + transfer[:3, 3])
                directions = _vectors(bank[selected], 'u') @ transfer[:3, :3].T
                moved = bank[selected].copy()
                _set_vectors(moved, 'r', points)
                _set_vectors(moved, 'u', directions)
                if (not np.isfinite(points).all()
                        or not np.isfinite(directions).all()):
                    raise ValueError('D1S coordinate transfer overflowed')
                bank[selected] = moved
                if isinstance(selector, Cells) and not has_dagmc:
                    _csg_points(model, selector, bank[selected & keep])
        if not keep.any():
            continue
        used.append({
            'files': tuple(str(f) for f in kernel.files),
            'source_weight': kernel.source_weight,
            'source_histories': kernel.source_histories,
            'parent_factors': dict(zip(kernel.parent_names, factors.tolist())),
        })
        mapping = []
        for name in kernel.parent_names:
            if name not in name_indices:
                name_indices[name] = len(names)
                names.append(name)
            mapping.append(name_indices[name])
        banks.append(bank[keep])
        weights.append(physical[keep])
        parents.append(
            np.asarray(mapping, dtype=np.int64)[kernel.parent_nuclide[keep]])
        component_ids.append(target_components[keep])
        kernel_ids.append(
            np.full(np.count_nonzero(keep), kernel_index, dtype=np.int64))
        for key in _METADATA:
            metadata[key].append(kernel.metadata[key][keep])
    if not banks:
        return 0., used, False
    bank = np.concatenate(banks)
    sampling_weights = np.concatenate(weights)
    total = fsum(sampling_weights)
    if not np.isfinite(total) or total <= 0.:
        raise ValueError('D1S photon source rate must be positive and finite')
    bank['wgt'] = 1.
    # An originating surface ID is not meaningful in a different geometry.
    bank['surf_id'] = 0
    if target.components is None and not has_dagmc:
        _csg_points(model, None, bank)
    if has_dagmc:
        for component in component_names:
            selector = target.components[component]
            if isinstance(selector, Cells):
                specs.append({'type': 'csg', 'pairs': selector._pairs})
            elif isinstance(selector, DAGMCVolumes):
                specs.append({
                    'type': 'dagmc', 'pairs': selector._pairs,
                    'universe': selector.universe.id,
                    'filename': str(_absolute(selector.universe.filename,
                                              target._base_dir)),
                })
            else:
                specs.append(None)
    version = contributions[0][0].version
    with h5py.File(filename, 'x') as handle:
        handle.attrs['filetype'] = np.bytes_('source')
        handle.attrs['version'] = version
        handle.attrs['d1s_version'] = 1
        handle.attrs['d1s_target_selectors'] = json.dumps(specs)
        handle.create_dataset('source_bank', data=bank)
        handle.create_dataset('sampling_weights', data=sampling_weights)
        handle.create_dataset(
            'parent_names', data=names, dtype=h5py.string_dtype())
        handle.create_dataset('parent_nuclide', data=np.concatenate(parents))
        handle.create_dataset(
            'component_names', data=component_names, dtype=h5py.string_dtype())
        handle.create_dataset(
            'target_component', data=np.concatenate(component_ids))
        handle.create_dataset('kernel_index', data=np.concatenate(kernel_ids))
        handle.create_dataset('photon_source_rate', data=total)
        for key in _METADATA:
            handle.create_dataset(key, data=np.concatenate(metadata[key]))
    return total, used, has_dagmc


def _dagmc_preflight():
    """Run only in an isolated Python process, never in the caller's C API."""
    import openmc.lib

    xml, filename = sys.argv[1:3]
    counts_only = filename == '--cell-counts'
    if not counts_only:
        with h5py.File(filename, 'r') as handle:
            bank = handle['source_bank'][()]
            components = handle['target_component'][()]
            specs = json.loads(handle.attrs['d1s_target_selectors'])
    # Plot initialization loads/places the geometry without transport/source
    # initialization or nuclear data. No mutation of the user's library occurs.
    try:
        openmc.lib.init(['-p', xml], output=False)
        if counts_only:
            counts = {uid: cell.num_instances
                      for uid, cell in openmc.lib.cells.items()}
            # Plot-mode initialization skips tally meshes. Load any unloaded
            # unstructured meshes explicitly in this isolated library session.
            model = openmc.Model.from_model_xml(xml)
            mesh_data = {}
            for tally in model.tallies:
                for filter_ in tally.filters:
                    mesh = getattr(filter_, 'mesh', None)
                    if (isinstance(mesh, openmc.UnstructuredMesh)
                            and mesh.id not in mesh_data):
                        # Mesh definitions can be loaded before plot-mode skips
                        # the tally filters. Reuse those runtime objects.
                        if mesh.id in openmc.lib.meshes:
                            lib_mesh = openmc.lib.meshes[mesh.id]
                        else:
                            lib_mesh = openmc.lib.Mesh.from_python(mesh)
                        mesh_data[mesh.id] = {
                            'n_elements': lib_mesh.n_elements,
                            'volumes': lib_mesh.volumes.tolist(),
                        }
            print('D1S_CELL_COUNTS:' + json.dumps(counts))
            print('D1S_MESH_DATA:' + json.dumps(mesh_data))
            return
        resolved = {}
        for spec in specs:
            if spec and spec['type'] == 'dagmc':
                uid = spec['universe']
                raw = _raw_volumes(spec['filename'])
                runtime = openmc.lib.dagmc_universe_cell_ids(uid)
                # DAGMC builds cells in entity-handle order. An implicit
                # complement absent from H5M is appended by DAGMC on loading.
                if len(runtime) not in (len(raw), len(raw) + 1):
                    raise RuntimeError(
                        'Cannot resolve raw DAGMC volume IDs for preflight')
                resolved[uid] = dict(zip(raw, map(int, runtime)))
        for point, component in zip(_vectors(bank, 'r'), components):
            cell, instance = openmc.lib.find_cell(point)
            if component < 0:
                continue  # All-sources mode: require containment only.
            spec = specs[component]
            if spec['type'] == 'csg':
                pairs = spec['pairs']
            else:
                pairs = [(resolved[spec['universe']][uid], inst)
                         for uid, inst in spec['pairs']]
            if not any(uid == cell.id and (inst is None or inst == instance)
                       for uid, inst in pairs):
                raise ValueError(
                    f'D1S point {point} is outside its target '
                    'component/instance')
    finally:
        if openmc.lib.is_initialized:
            openmc.lib.finalize()


def _preflight(model, directory, bank=None, mesh_data=False):
    locator_model = deepcopy(model)
    # Plot initialization needs only geometry. Do not initialize the replay
    # FileSource (or require its parent chain) before the locator can run.
    locator_model.settings.source = openmc.IndependentSource()
    locator_model.settings._surf_source_read = None
    locator_model.settings.use_decay_photons = False
    locator_model.settings.d1s_source_write = False
    # Capture forces leaf offsets; the isolated locator needs the same offsets
    # even when the user disabled them and no instance filter is present.
    locator_model.settings.material_cell_offsets = True
    if not mesh_data:
        locator_model.tallies = []
    locator_model.plots = []
    locator_model.export_to_model_xml(directory / 'model.xml')
    environment = os.environ.copy()
    # Model.run changes cwd. Source-tree/editable imports must work in the
    # isolated locator too, without depending on its working directory.
    environment['PYTHONPATH'] = os.pathsep.join(
        str(Path(p or '.').resolve()) for p in sys.path)
    result = subprocess.run(
        [sys.executable, '-c',
         'from openmc.deplete._d1s import _dagmc_preflight; '
         '_dagmc_preflight()', str(directory / 'model.xml'),
         str(bank) if bank is not None else '--cell-counts'],
        cwd=directory, env=environment, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(
            'D1S target geometry preflight failed. A compatible Python '
            'shared library with the required geometry/mesh features '
            'is required.\n'
            + result.stdout + result.stderr)
    if bank is None:
        counts = None
        meshes = {}
        for line in result.stdout.splitlines():
            if line.startswith('D1S_CELL_COUNTS:'):
                counts = {
                    int(uid): count for uid, count in
                    json.loads(line.removeprefix('D1S_CELL_COUNTS:')).items()}
            elif line.startswith('D1S_MESH_DATA:'):
                meshes = {
                    int(uid): data for uid, data in
                    json.loads(line.removeprefix('D1S_MESH_DATA:')).items()}
        if counts is None:
            raise RuntimeError(
                'D1S geometry locator did not return cell counts')
        return (counts, meshes) if mesh_data else counts


def _tallies(model, statepoint, rate, labels, cell_counts=None,
            mesh_data=None):
    if statepoint is None:
        model.geometry.determine_paths()
        cells = model.geometry.get_all_cells()
        tallies = deepcopy(list(model.tallies))
        for tally in tallies:
            # Backend statepoints label the implicit nuclide bin as 'total'.
            # Match that schema so DataFrame and slicing helpers work at zero.
            if not tally.nuclides:
                tally.nuclides = ['total']
            for filter_ in tally.filters:
                mesh = getattr(filter_, 'mesh', None)
                if (isinstance(mesh, openmc.UnstructuredMesh)
                        and mesh_data is not None):
                    data = mesh_data[mesh.id]
                    mesh.n_elements = data['n_elements']
                    mesh.volumes = data['volumes']
                    mesh._has_statepoint_data = True
                    # Refresh bins now that the backend supplied the mesh size.
                    filter_.mesh = mesh
                if isinstance(filter_, openmc.DistribcellFilter):
                    uid = int(filter_.bins[0])
                    if cell_counts is not None:
                        filter_._num_bins = cell_counts[uid]
                        filter_._paths = None
                    else:
                        cell = cells[uid]
                        filter_._num_bins = cell.num_instances
                        filter_._paths = cell.paths
            tally._sp_filename = None
            tally._derived = True
            tally._sparse = False
            for key in ('_mean', '_std_dev', '_sum', '_sum_sq'):
                setattr(tally, key, np.zeros(tally.shape))
            tally._num_realizations = 0
            tally._results_read = True
            tally._vov = None
            if tally.higher_moments:
                tally._sum_third = np.zeros(tally.shape)
                tally._sum_fourth = np.zeros(tally.shape)
    else:
        with openmc.StatePoint(statepoint, autolink=False) as sp:
            tallies = []
            for original in sp.tallies.values():
                # Materialize all results before closing the raw statepoint.
                original.mean, original.std_dev, original.sum, original.sum_sq
                tally = deepcopy(original)
                tally._sparse = False
                tally._mean = original.mean * rate
                tally._std_dev = original.std_dev * rate
                tally._sum = original.sum * rate
                tally._sum_sq = original.sum_sq * rate**2
                tally._vov = None
                if original.higher_moments:
                    tally._sum_third = original.sum_third * rate**3
                    tally._sum_fourth = original.sum_fourth * rate**4
                tally._results_read = True
                tallies.append(tally)
    for tally in tallies:
        tally.d1s_metadata = dict(
            labels, normalization='physical rate', uncertainty=_UNCERTAINTY)
    return MappingProxyType({t.id: t for t in tallies})


class Simulation:
    """A multi-model D1S irradiation/cooling schedule, without depletion.

    Parameters
    ----------
    steps : iterable of Irradiation or Cooling
        Ordered nonempty schedule. Step names are unique across both kinds.
        Global time is the sum of all preceding durations plus local time.
        Basic schedule/mapping checks run immediately; :meth:`validate` also
        checks geometries, selector membership and transport settings.
    chain_file : path-like, optional
        Decay/reaction data XML file for backend photon production. Defaults to
        openmc.config['chain_file']; resolved absolutely before changing cwd.
        It is installed with config.patch for each backend run. This supplies
        decay data only; no depletion calculation is performed.
    """

    def __init__(self, steps, chain_file=None):
        cv.check_type('steps', steps, Iterable)
        self.steps = tuple(steps)
        if not self.steps:
            raise ValueError('D1S requires at least one schedule step')
        for step in self.steps:
            cv.check_type('step', step, (Irradiation, Cooling))
        counts, names = {'irradiation': 0, 'cooling': 0}, []
        for step in self.steps:
            kind = ('irradiation' if isinstance(step, Irradiation)
                    else 'cooling')
            counts[kind] += 1
            names.append(
                step.name if step.name is not None
                else f'{kind}-{counts[kind]}')
        if len(set(names)) != len(names):
            raise ValueError(
                'D1S step names must be unique (including generated names)')
        self.step_names = tuple(names)
        path = (chain_file if chain_file is not None
                else openmc.config.get('chain_file'))
        if path is not None:
            cv.check_type('chain_file', path, cv.PathLike)
        self.chain_file = (
            None if path is None else Path(path).expanduser().resolve())
        self._check_schedule()

    def _check_schedule(self):
        modes = {step.configuration.components is None for step in self.steps}
        if len(modes) > 1:
            raise ValueError(
                'No-map configurations can only pair with other no-map '
                'configurations')
        historical = set()
        elapsed = 0.
        for step in self.steps:
            cv.check_type('configuration', step.configuration, Configuration)
            _seconds(step.duration, 'duration', positive=True)
            configuration = step.configuration
            if configuration.components is not None:
                missing = historical - configuration.components.keys()
                if missing:
                    raise ValueError(
                        f'Target configuration is missing historical '
                        f'identities '
                        f'{sorted(missing)}; use explicit None for removal')
            if isinstance(step, Irradiation):
                _real('source_rate', step.source_rate)
                if step.source_rate and configuration.components is not None:
                    historical.update(
                        key for key, value in configuration.components.items()
                        if value is not None)
            else:
                if len(set(step.sample_times)) != len(step.sample_times):
                    raise ValueError('Cooling sample times must be unique')
                for time in step.sample_times:
                    if _seconds(time) > step.duration:
                        raise ValueError(
                            'Cooling sample times must lie in [0, duration]')
            elapsed = _real('global schedule time', elapsed + step.duration)

    def validate(self):
        """Validate the complete schedule without changing models or library
        state.

        Raises for invalid CE/fixed-source settings, selectors not in models,
        overlapping component selectors, invalid raw CAD volume IDs/instances,
        a missing explicit chain path, or an already initialized global C API.
        Backend feature availability is checked only when transport is needed.
        Returns None. This method does not run neutron/photon transport.
        """
        self._check_schedule()
        _uninitialized()
        if self.chain_file is not None and not self.chain_file.is_file():
            raise FileNotFoundError(self.chain_file)
        if self.chain_file is not None:
            from .chain import _get_chain
            _get_chain(self.chain_file)
        for configuration in dict.fromkeys(
            step.configuration for step in self.steps
        ):
            _validate_configuration(configuration)

    def prepare(self, output_dir='d1s-production', run_kwargs=None):
        """Capture one bank per distinct positive-rate irradiation
        Configuration.

        Parameters
        ----------
        output_dir : path-like, optional
            New or empty dedicated directory. Each kernel gets a new numbered
            subdirectory; existing nonempty directories are never overwritten.
        run_kwargs : mapping, optional
            Model.run options: particles, threads, geometry_debug, tracks,
            output, openmc_exec, mpi_args, event_based. Input/output locations,
            restarts, exports and applying results to caller models are owned
            by D1S and cannot be overridden.

        Returns
        -------
        Production
            Banks normalized by the sum of source_histories over ALL MPI
            shards, including killed histories and ranks with no photons.
            Rates and durations are not cached.
        """
        self.validate()
        options = _run_options(run_kwargs)
        configurations = tuple(dict.fromkeys(
            step.configuration for step in self.steps
            if isinstance(step, Irradiation) and step.source_rate))
        if configurations:
            if self.chain_file is None:
                raise ValueError(
                    'D1S photon production requires chain_file or '
                    'openmc.config["chain_file"]')
            _capture_setting(deepcopy(configurations[0].model.settings), True)
        root = _new_directory(output_dir)
        kernels = {}
        for i, configuration in enumerate(configurations):
            directory = root / f'kernel-{i:04d}'
            directory.mkdir()
            model = _clone(configuration)
            statepoint = _transport(
                model, directory, options, self.chain_file, True)
            files = tuple(sorted(directory.glob('d1s_source.*.h5'),
                                 key=lambda p: int(p.name.split('.')[1])))
            if not files:
                raise RuntimeError(
                    'D1S capture produced no rank shards; '
                    'check backend support')
            ranks = [int(path.name.split('.')[1]) for path in files]
            if ranks != list(range(len(files))):
                raise RuntimeError('D1S capture rank shards are incomplete')
            mpi_args = options.get('mpi_args') or []
            for flag in ('-n', '-np', '--np', '--ntasks'):
                if (flag in mpi_args
                        and len(files) != int(
                            mpi_args[mpi_args.index(flag) + 1])):
                    raise RuntimeError(
                        'Missing D1S MPI shards, including possibly empty '
                        'ranks')
            kernels[configuration] = _read_kernel(
                configuration, files, statepoint)
        return Production(kernels, root, self.chain_file)

    def evaluate(self, production, output_dir='d1s-evaluation',
                 run_kwargs=None):
        """Weight prepared banks for this schedule and run cooling
        observations.

        Parameters
        ----------
        production : Production
            Data from prepare, using the SAME Configuration objects for every
            positive-rate irradiation state. Additional prepared states may be
            unused. Rates, durations, cooling models and sample times may vary.
        output_dir : path-like, optional
            New or empty directory, with dedicated numbered snapshot
            directories.
        run_kwargs : mapping, optional
            Same controlled subprocess transport options as prepare.

        Returns
        -------
        Results
            Physical-rate tallies, observation metadata and raw statepoint
            paths (None when transport was skipped for zero source). Histories
            survive removal and reinsertion. Only completed irradiation
            intervals preceding a cooling observation contribute.
        """
        cv.check_type('production', production, Production)
        self.validate()
        options = _run_options(run_kwargs)
        if (self.chain_file is not None
                and self.chain_file != production.chain_file):
            raise ValueError(
                'Production and evaluation must use the same decay-data file')
        required = {step.configuration for step in self.steps
                    if isinstance(step, Irradiation) and step.source_rate}
        if not required <= production.kernels.keys():
            raise ValueError(
                'Production lacks an irradiation Configuration; '
                'reuse the same objects')
        root = _new_directory(output_dir)
        intervals, observations, elapsed = [], {}, 0.
        decay = {}
        for index, (name, step) in enumerate(zip(self.step_names, self.steps)):
            if isinstance(step, Irradiation):
                if step.source_rate:
                    intervals.append((step, elapsed + step.duration))
            else:
                for local in step.sample_times:
                    global_time = elapsed + local
                    factors = {}
                    for interval, end in intervals:
                        configuration = interval.configuration
                        kernel = production.kernels[configuration]
                        if configuration not in decay:
                            decay[configuration] = np.array([
                                openmc.data.decay_constant(
                                    parent, chain_file=production.chain_file)
                                for parent in kernel.parent_names])
                            if (not np.isfinite(decay[configuration]).all()
                                    or (decay[configuration] < 0.).any()):
                                raise ValueError(
                                    'Parent decay constants must be finite '
                                    'and nonnegative')
                        lam = decay[configuration]
                        factor = (interval.source_rate
                                  * -np.expm1(-lam * interval.duration)
                                  * np.exp(-lam * (global_time - end)))
                        factors[configuration] = (
                            factors.get(configuration, np.zeros_like(lam))
                            + factor)
                    contributions = [(production.kernels[c], f)
                                     for c, f in factors.items()]
                    directory = root / f'snapshot-{len(observations):04d}'
                    directory.mkdir()
                    model = _clone(step.configuration)
                    bank = directory / 'photons.h5'
                    rate, used, cad = _compose(
                        step.configuration, model, contributions, bank)
                    statepoint = None
                    if rate > 0.:
                        # Fixed-source tallies include the sum of source
                        # strengths. Keep backend responses per photon and
                        # apply the physical photon rate exactly once when
                        # materializing the tallies.
                        model.settings.source = openmc.FileSource(
                            bank, strength=1.)
                        # This legacy setting appends another external
                        # source in the backend even after Settings.source
                        # is replaced.
                        model.settings._surf_source_read = None
                        model.settings.uniform_source_sampling = False
                        model.settings.photon_transport = True
                        model.settings.use_decay_photons = False
                        _capture_setting(model.settings, False)
                        if cad:
                            _preflight(model, directory, bank)
                        statepoint = _transport(
                            model, directory, options,
                            production.chain_file, False)
                        if statepoint is None:
                            raise RuntimeError(
                                'D1S photon transport did not write a '
                                'statepoint')
                    cell_counts = None
                    meshes = None
                    if rate == 0.:
                        unstructured = any(
                            isinstance(getattr(f, 'mesh', None),
                                       openmc.UnstructuredMesh)
                            for t in model.tallies for f in t.filters)
                        cad_distribcell = (
                            any(isinstance(u, openmc.DAGMCUniverse)
                                for u in
                                model.geometry.get_all_universes().values())
                            and any(t.contains_filter(openmc.DistribcellFilter)
                                    for t in model.tallies))
                        if unstructured:
                            cell_counts, meshes = _preflight(
                                model, directory, mesh_data=True)
                        elif cad_distribcell:
                            cell_counts = _preflight(model, directory)
                    labels = {
                        'step': name, 'step_index': index, 'local_time': local,
                        'global_time': global_time, 'source_rate': rate,
                        'time_units': 's',
                    }
                    tallies = _tallies(
                        model, statepoint, rate, labels, cell_counts, meshes)
                    provenance = MappingProxyType({
                        'kernels': tuple(used),
                        'target_validation': (
                            'isolated C API' if cad else 'CSG find'),
                        'uncertainty': _UNCERTAINTY,
                    })
                    observations[name, local] = Observation(
                        name, index, local, global_time, rate, statepoint,
                        bank if rate else None, tallies, provenance)
            elapsed += step.duration
        return Results(observations, self.step_names, production)

    def run(self, output_dir='d1s', run_kwargs=None):
        """Prepare production then evaluate, returning :class:`Results`.

        ``output_dir`` must be new or empty; production and observations get
        separate dedicated subdirectories. ``run_kwargs`` is as for prepare.
        No caller model, tally schema, in-memory C API model, or raw statepoint
        is modified. Bank sampling uncertainty is not propagated into std_dev.
        """
        self.validate()
        _run_options(run_kwargs)
        root = _new_directory(output_dir)
        production = self.prepare(root / 'production', run_kwargs)
        return self.evaluate(production, root / 'observations', run_kwargs)