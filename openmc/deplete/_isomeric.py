"""Support for extracting energy-dependent isomeric production data.

This module reads ENDF MF=8/9/10 radioactive nuclide production sections
from incident neutron evaluations and maps the final-state level numbers
(LFS) found there to metastable nuclide names by matching excitation
energies against decay data. The tabulated MF=9 yields and MF=10 partial
cross sections are kept verbatim; no arithmetic is performed on them.

Note that an LFS value is a level index of the product nuclide, not a
metastable-state index, so targets are never named directly from it.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO

import numpy as np

from openmc.data import Tabulated1D, gnds_name, zam
import openmc.data.endf as endf6

from .nuclide import IsomericProduction, ProductionTable

# Assignment status values used in the mapping report
MATCHED = 'matched'
POSITIONAL = 'positional'
GROUND = 'ground'
FOLDED = 'folded'

# Final states with an excitation energy below this value [eV] are treated
# as the ground state regardless of their level number
GROUND_ENERGY_CUTOFF = 1.0

THERMAL_ENERGY = 0.0253


@dataclass
class LevelRecord:
    """Raw MF=9 or MF=10 data for the production of one final state."""

    mf: int
    mt: int
    zap: int                  # ZA of the product; may be 0 when not given
    lfs: int                  # ENDF level number of the final state
    elfs: float | None        # MF=8 excitation energy [eV]; None when absent
    qm: float                 # [eV]
    qi: float                 # [eV]
    data: Tabulated1D
    source: str               # library label, e.g. 'ENDF/B-8.1'

    @property
    def excitation_energy(self):
        """Excitation energy [eV], preferring the MF=8 ELFS value."""
        if self.elfs is not None:
            return self.elfs
        return self.qm - self.qi


@dataclass
class LevelAssignment:
    """The chain target chosen for one level record."""

    record: LevelRecord
    target: str
    status: str
    liso: int | None = None
    decay_elis: float | None = None


def library_label(evaluation):
    """Return a short provenance label for an evaluation.

    Parameters
    ----------
    evaluation : openmc.data.endf.Evaluation
        Evaluation to label

    Returns
    -------
    str
        Label such as 'ENDF/B-8.1' or 'TENDL-2025.0'

    """
    library, version, release = evaluation.info['library']
    return f'{library}-{version}.{release}'


def extract_isomeric_production(evaluation, source=None):
    """Read MF=8/9/10 production sections from an evaluation.

    MF=8 subsections that point at other files for the production data
    (LMF not equal to 9 or 10, e.g. the LMF=6 product distributions found
    on TENDL MT=5) are skipped.

    Parameters
    ----------
    evaluation : openmc.data.endf.Evaluation
        Incident neutron evaluation to read
    source : str, optional
        Provenance label stored on each record. Defaults to
        :func:`library_label` of the evaluation.

    Returns
    -------
    dict
        Mapping of MT numbers to lists of :class:`LevelRecord`

    """
    if source is None:
        source = library_label(evaluation)

    by_mt = defaultdict(set)
    for mf, mt in evaluation.section:
        if mf in (9, 10):
            by_mt[mt].add(mf)

    result = {}
    for mt in sorted(by_mt):
        # MF=8 links each (ZAP, LFS) pair to an excitation energy and says
        # in which file the production data lives (LMF)
        mf8_info = {}
        if (8, mt) in evaluation.section:
            file_obj = StringIO(evaluation.section[8, mt])
            items = endf6.get_head_record(file_obj)
            n_states, complete_flag = items[4], items[5]
            for _ in range(n_states):
                if complete_flag == 0:
                    sub, _values = endf6.get_list_record(file_obj)
                else:
                    sub = endf6.get_cont_record(file_obj)
                zap, elfs = int(sub[0]), float(sub[1])
                lmf, lfs = int(sub[2]), int(sub[3])
                mf8_info[zap, lfs] = (elfs, lmf)

        levels = []
        for mf in sorted(by_mt[mt]):
            file_obj = StringIO(evaluation.section[mf, mt])
            items = endf6.get_head_record(file_obj)
            n_states = items[4]
            for _ in range(n_states):
                params, func = endf6.get_tab1_record(file_obj)
                qm, qi = params[0], params[1]
                zap, lfs = int(params[2]), int(params[3])
                elfs, lmf = mf8_info.get((zap, lfs), (None, None))
                if lmf is not None and lmf not in (9, 10):
                    continue
                levels.append(LevelRecord(
                    mf=mf, mt=mt, zap=zap, lfs=lfs, elfs=elfs,
                    qm=qm, qi=qi, data=func, source=source))
        if levels:
            result[mt] = levels
    return result


def assign_levels(records, ground_daughter, isomer_energies, elis_rtol):
    """Assign the level records of one reaction to chain target names.

    Levels are first matched to metastable states by excitation energy
    (nearest decay-library state within ``elis_rtol`` relative tolerance).
    Levels that fail the energy match are paired positionally with the
    remaining metastables in order of increasing excitation energy, and
    anything left over is folded onto the ground-state target.

    Parameters
    ----------
    records : list of LevelRecord
        Level records for a single transmutation reaction
    ground_daughter : str
        Ground-state product from mass and charge arithmetic, used when a
        record does not carry a product ZA
    isomer_energies : dict
        Mapping ``{(Z, A): [(liso, elis), ...]}`` of metastable states with
        positive excitation energies from the decay data
    elis_rtol : float
        Relative tolerance on the excitation energy match

    Returns
    -------
    list of LevelAssignment

    """
    assignments = [None] * len(records)
    unmatched = []
    matched_liso = defaultdict(set)

    for i, record in enumerate(records):
        if record.zap > 0:
            z, a = divmod(record.zap, 1000)
        else:
            z, a, _ = zam(ground_daughter)
        target_elis = record.excitation_energy
        if record.lfs == 0 or target_elis < GROUND_ENERGY_CUTOFF:
            assignments[i] = LevelAssignment(
                record, gnds_name(z, a, 0), GROUND)
            continue

        best = None
        for liso, elis in isomer_energies.get((z, a), []):
            diff = abs(target_elis - elis)
            if diff <= elis_rtol * elis and (best is None or diff < best[2]):
                best = (liso, elis, diff)
        if best is not None:
            liso, elis, _diff = best
            assignments[i] = LevelAssignment(
                record, gnds_name(z, a, liso), MATCHED, liso, elis)
            matched_liso[z, a].add(liso)
        else:
            unmatched.append((i, record, z, a, target_elis))

    # Positional fallback per product nuclide. Pairing is done per LEVEL,
    # not per record, since one level may carry both an MF=9 and an MF=10
    # record and both must land on the same target.
    by_za_level = defaultdict(lambda: defaultdict(list))
    for i, record, z, a, target_elis in unmatched:
        by_za_level[z, a][record.lfs].append((i, record, target_elis))
    for (z, a), levels in by_za_level.items():
        available = [
            (liso, elis)
            for liso, elis in sorted(isomer_energies.get((z, a), []))
            if liso not in matched_liso[z, a]]
        ordered = sorted(levels.values(), key=lambda entries: entries[0][2])
        for entries, (liso, elis) in zip(ordered, available):
            for i, record, _elis in entries:
                assignments[i] = LevelAssignment(
                    record, gnds_name(z, a, liso), POSITIONAL, liso, elis)
        for entries in ordered[len(available):]:
            for i, record, _elis in entries:
                assignments[i] = LevelAssignment(
                    record, gnds_name(z, a, 0), FOLDED)

    return assignments


def group_records(records_and_targets):
    """Group level records into IsomericProduction objects per target.

    Records with the same level number attached to the same target (an
    MF=9 and an MF=10 table for one state) share one
    :class:`~openmc.deplete.IsomericProduction` instance.

    Parameters
    ----------
    records_and_targets : list of (LevelRecord, str)
        Level records with the final chain target each is attached to

    Returns
    -------
    dict
        Mapping of target names to lists of IsomericProduction

    """
    result = defaultdict(list)
    index = {}
    for record, target in records_and_targets:
        key = (target, record.lfs)
        production = index.get(key)
        if production is None:
            elfs = record.excitation_energy
            if record.lfs == 0:
                elfs = max(elfs, 0.0)
            production = IsomericProduction(record.lfs, elfs, [])
            index[key] = production
            result[target].append(production)
        production.tables.append(ProductionTable(
            mf=record.mf, mt=record.mt, source=record.source,
            QM=record.qm, QI=record.qi, data=record.data))
    return dict(result)


def select_records(mts, primary, supplement):
    """Choose the level records for one transmutation reaction.

    The primary library wins over the supplement. Within a library, only
    the first MT with data (in ascending order, so a summary section is
    preferred over any partials) is used so that a summary section and
    its partials cannot be double counted; any further MTs carrying data
    are returned so the caller can flag them.

    Parameters
    ----------
    mts : set of int
        Candidate ENDF MT numbers of the reaction
    primary : dict
        Mapping ``{mt: [LevelRecord]}`` from the primary evaluation
    supplement : dict
        Mapping ``{mt: [LevelRecord]}`` from supplementary evaluations

    Returns
    -------
    records : list of LevelRecord
        Records of the first MT with data, or an empty list
    skipped_mts : list of int
        Additional MTs that also carry data but were not used

    """
    available = primary if (mts & set(primary)) else supplement
    with_data = sorted(mts & set(available))
    if not with_data:
        return [], []
    return available[with_data[0]], with_data[1:]


def _collapse(table, mode):
    """Collapse an MF=9 yield table to a scalar for the requested mode.

    Returns None when the thermal mode is requested but the table's grid
    does not cover 0.0253 eV. In multigroup mode the group midpoints are
    clamped into the tabulated range, so every group contributes using
    the table's edge values.
    """
    x = table.data.x
    if mode == 'thermal':
        if x[0] > THERMAL_ENERGY or x[-1] < THERMAL_ENERGY:
            return None
        return float(table.data(THERMAL_ENERGY))

    energies, flux = mode
    energies = np.asarray(energies, dtype=float)
    flux = np.asarray(flux, dtype=float)
    # Group representative energies: geometric midpoints of the group
    # boundaries, clamped into the tabulated range so that groups
    # straddling or outside the grid (e.g. a first group starting at
    # 0 eV) use the table's edge values instead of contributing zero
    midpoints = np.sqrt(energies[:-1] * energies[1:])
    midpoints = np.clip(midpoints, x[0], x[-1])
    values = table.data(midpoints)
    return float(values @ flux / flux.sum())


def compute_scalar_ratios(productions_by_target, ground_target, mode):
    """Compute scalar branching ratios per target for one reaction.

    The default mode ``'none'`` reproduces the historical behavior: the
    full reaction rate goes to the ground-state target. The ``'thermal'``
    mode evaluates the MF=9 yields at 0.0253 eV, and an ``(energies,
    flux)`` tuple collapses them with a multigroup flux, evaluated at the
    geometric midpoints of the group boundaries clamped into the
    tabulated range. The ground-state share is always taken as one minus
    the metastable sum, which also covers evaluations where the ground
    yield is implicit. When the requested mode cannot be supported by
    the stored data (an MF=10-only state, a thermal request outside the
    grid, or metastable yields summing above one), the ratios fall back
    to the ``'none'`` values and a note is returned.

    Parameters
    ----------
    productions_by_target : dict
        Mapping of target names to lists of IsomericProduction
    ground_target : str
        Name of the ground-state target
    mode : {'none', 'thermal'} or tuple
        Scalar branching mode

    Returns
    -------
    ratios : dict
        Mapping of target names to scalar branching ratios summing to one
    note : str or None
        Explanation when the ratios fell back to the default

    """
    default = {target: 0.0 for target in productions_by_target}
    default[ground_target] = 1.0
    if mode == 'none' or set(default) == {ground_target}:
        return default, None

    ratios = dict(default)
    meta_sum = 0.0
    for target, productions in productions_by_target.items():
        if target == ground_target:
            continue
        value = 0.0
        for production in productions:
            table = next(
                (t for t in production.tables if t.mf == 9), None)
            if table is None:
                return default, (
                    f'no MF=9 yield for {target}; scalar branching ratios '
                    'left at the default')
            collapsed = _collapse(table, mode)
            if collapsed is None:
                return default, (
                    f'MF=9 grid for {target} does not cover 0.0253 eV; '
                    'scalar branching ratios left at the default')
            value += collapsed
        ratios[target] = value
        meta_sum += value

    if meta_sum > 1.0:
        return default, (
            'metastable MF=9 yields sum above one; scalar branching '
            'ratios left at the default')
    ratios[ground_target] = 1.0 - meta_sum
    return ratios, None


class IsomerMappingReport:
    """Collects level-to-isomer mapping decisions made during a chain build.

    Parameters
    ----------
    elis_rtol : float
        Relative tolerance that was used for excitation energy matching

    """

    def __init__(self, elis_rtol):
        self.elis_rtol = elis_rtol
        self.rows = []
        self.notes = []

    def add(self, parent, reaction, assignment, target, status, note=''):
        record = assignment.record
        self.rows.append({
            'parent': parent,
            'reaction': reaction,
            'mt': record.mt,
            'mf': record.mf,
            'lfs': record.lfs,
            'elfs': record.excitation_energy,
            'target': target,
            'status': status,
            'liso': assignment.liso,
            'decay_elis': assignment.decay_elis,
            'source': record.source,
            'note': note,
        })

    def add_note(self, parent, reaction, text):
        self.notes.append((parent, reaction, text))

    @property
    def counts(self):
        return Counter(row['status'] for row in self.rows)

    def write(self, path):
        """Write a human readable mapping report.

        Parameters
        ----------
        path : str or os.PathLike
            File to write the report to

        """
        columns = ('parent', 'reaction', 'mt', 'mf', 'lfs', 'elfs',
                   'target', 'status', 'liso', 'decay_elis', 'source',
                   'note')
        header = ('ISOMER MAPPING REPORT\n'
                  f'ELIS matching relative tolerance: {self.elis_rtol}\n')
        counts = self.counts
        summary = ['Level records: {}'.format(sum(counts.values()))]
        for status in (MATCHED, POSITIONAL, GROUND, FOLDED):
            summary.append(f'  {status}: {counts.get(status, 0)}')

        lines = [header, '\n'.join(summary), '']
        lines.append('  '.join(f'{c:>12}' for c in columns))
        for row in self.rows:
            formatted = []
            for column in columns:
                value = row[column]
                if value is None:
                    value = '-'
                elif isinstance(value, float):
                    value = f'{value:.6g}'
                formatted.append(f'{value!s:>12}')
            lines.append('  '.join(formatted))
        if self.notes:
            lines.append('')
            lines.append('NOTES')
            for parent, reaction, text in self.notes:
                lines.append(f'  {parent} {reaction}: {text}')
        with open(path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
