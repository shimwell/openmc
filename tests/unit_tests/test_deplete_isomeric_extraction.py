"""Tests for ENDF MF=8/9/10 extraction and level-to-isomer mapping.

The fixture strings below are verbatim section text from the
ENDF/B-VIII.1 Am241 and Nb93 incident neutron evaluations.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from openmc.deplete import Chain
from openmc.deplete._isomeric import (
    FOLDED, GROUND, MATCHED, POSITIONAL, IsomerMappingReport, assign_levels,
    compute_scalar_ratios, extract_isomeric_production, group_records,
    select_records)

AM241_MF8_MT102 = """\
 9.524100+4 2.389860+2          0          0          2          09543 8102    1
 9.524201+4 0.000000+0          9          0         24          09543 8102    2
 5.767200+4 1.400000+0 9.624201+4 3.686630-1 6.648200+5 2.010000+09543 8102    3
 5.767200+4 1.400000+0 9.624201+4 4.583370-1 6.226900+5 2.020000+09543 8102    4
 5.767200+4 2.400000+0 9.424201+4 6.700000-2 7.509630+5 2.010000+09543 8102    5
 5.767200+4 2.400000+0 9.424201+4 1.060000-1 7.064230+5 2.020000+09543 8102    6
 9.524201+4 4.863000+4          9          2        120          09543 8102    7
 4.446580+9 3.100000+0 9.524201+4 8.232040-1 4.863000+4 2.000000+09543 8102    8
 4.446580+9 3.200000+0 9.524201+4 1.722060-1 4.863000+4 2.010000+09543 8102    9
 4.446580+9 4.100000+0 9.323801+4 2.747740-7 5.561940+6 2.020000+09543 8102   10
 4.446580+9 4.100000+0 9.323801+4 6.411380-6 5.501660+6 2.040000+09543 8102   11
 4.446580+9 4.100000+0 9.323801+4 4.762740-5 5.452290+6 2.070000+09543 8102   12
 4.446580+9 4.100000+0 9.323801+4 5.358080-5 5.409190+6 2.090000+09543 8102   13
 4.446580+9 4.100000+0 9.323801+4 3.159900-5 5.355540+6 2.110000+09543 8102   14
 4.446580+9 4.100000+0 9.323801+4 3.938420-5 5.312840+6 2.130000+09543 8102   15
 4.446580+9 4.100000+0 9.323801+4 1.831820-6 5.290040+6 2.140000+09543 8102   16
 4.446580+9 4.100000+0 9.323801+4 5.037510-6 5.287640+6 2.150000+09543 8102   17
 4.446580+9 4.100000+0 9.323801+4 1.373870-6 5.254040+6 2.160000+09543 8102   18
 4.446580+9 4.100000+0 9.323801+4 4.114280-3 5.245940+6 2.170000+09543 8102   19
 4.446580+9 4.100000+0 9.323801+4 9.159120-7 5.191740+6 2.200000+09543 8102   20
 4.446580+9 4.100000+0 9.323801+4 2.665300-4 5.179840+6 2.210000+09543 8102   21
 4.446580+9 4.100000+0 9.323801+4 8.701160-6 5.125740+6 2.220000+09543 8102   22
 4.446580+9 4.100000+0 9.323801+4 1.373870-6 5.119740+6 2.230000+09543 8102   23
 4.446580+9 4.100000+0 9.323801+4 1.007500-5 5.101140+6 2.240000+09543 8102   24
 4.446580+9 4.100000+0 9.323801+4 9.159120-7 5.063440+6 2.250000+09543 8102   25
 4.446580+9 4.100000+0 9.323801+4 9.159120-8 5.010340+6 2.260000+09543 8102   26
 4.446580+9 6.000000+0 0.000000+0 4.70000-11 0.000000+0 1.000000+09543 8102   27
"""

AM241_MF9_MT102 = """\
 9.524100+4 2.389860+2          0          0          2          09543 9102    1
 5.537755+6 5.537755+6      95242          0          1          99543 9102    2
          9          3                                            9543 9102    3
 1.000000-5 9.000000-1 3.690000-1 9.000000-1 1.000000+3 8.667000-19543 9102    4
 1.000000+5 8.420000-1 6.000001+5 8.153300-1 1.000000+6 7.438200-19543 9102    5
 2.000000+6 5.703000-1 4.000001+6 5.200000-1 3.000000+7 5.200000-19543 9102    6
 5.537755+6 5.489125+6      95242          2          1          99543 9102    7
          9          3                                            9543 9102    8
 1.000000-5 1.000000-1 3.690000-1 1.000000-1 1.000000+3 1.333000-19543 9102    9
 1.000000+5 1.580000-1 6.000001+5 1.846700-1 1.000000+6 2.561800-19543 9102   10
 2.000000+6 4.297000-1 4.000001+6 4.800000-1 3.000000+7 4.800000-19543 9102   11
"""

NB93_MF8_MT4 = """\
 4.109300+4 9.210510+1          0          0          1          14125 8  4    1
 4.109300+4 3.073000+4         10          1          0          04125 8  4    2
"""

NB93_MF10_MT4 = """\
 4.109300+4 9.210510+1          0          0          1          0412510  4    1
 0.000000+0-3.073000+4      41093          1          1         34412510  4    2
         34          2          0          0          0          0412510  4    3
 3.115000+4 0.000000+0 4.000000+4 6.053000-5 5.000000+4 1.513000-4412510  4    4
 6.000000+4 2.522000-4 8.000000+4 5.246000-4 1.000000+5 8.676000-4412510  4    5
 1.500000+5 1.987000-3 2.000000+5 3.460000-3 3.000000+5 7.253000-3412510  4    6
 4.000000+5 1.198000-2 5.000000+5 1.742000-2 6.000000+5 2.337000-2412510  4    7
 7.000000+5 2.921000-2 7.625000+5 4.035000-2 1.000000+6 7.484000-2412510  4    8
 1.288000+6 1.122000-1 1.750000+6 2.110000-1 2.063000+6 2.060000-1412510  4    9
 2.563000+6 2.486000-1 4.050000+6 2.782000-1 4.588000+6 2.629000-1412510  4   10
 6.156000+6 2.455000-1 8.344000+6 2.446000-1 9.375000+6 2.089000-1412510  4   11
 1.037500+7 1.349000-1 1.139400+7 8.441000-2 1.239400+7 5.764000-2412510  4   12
 1.340600+7 4.385000-2 1.435600+7 3.756000-2 1.552000+7 3.088000-2412510  4   13
 1.700000+7 2.718000-2 2.000000+7 2.265000-2 2.000000+7 0.000000+0412510  4   14
 1.500000+8 0.000000+0                                            412510  4   15
"""


class StubEvaluation:
    """Minimal object exposing the raw section text of an evaluation."""

    def __init__(self, sections):
        self.section = sections


@pytest.fixture
def am241_records():
    stub = StubEvaluation({
        (8, 102): AM241_MF8_MT102,
        (9, 102): AM241_MF9_MT102,
    })
    return extract_isomeric_production(stub, source='ENDF/B-8.1')


def test_extract_am241(am241_records):
    assert set(am241_records) == {102}
    ground, meta = am241_records[102]

    assert ground.mf == 9
    assert ground.zap == 95242
    assert ground.lfs == 0
    assert ground.elfs == 0.0
    assert ground.qm == 5537755.0
    assert ground.qi == 5537755.0
    assert ground.source == 'ENDF/B-8.1'

    assert meta.lfs == 2
    assert meta.elfs == 48630.0
    assert meta.qi == 5489125.0
    assert meta.excitation_energy == 48630.0

    # Verbatim 9-point single-region lin-log tables
    np.testing.assert_array_equal(ground.data.breakpoints, [9])
    np.testing.assert_array_equal(ground.data.interpolation, [3])
    assert len(meta.data.x) == 9
    assert ground.data(0.0253) == pytest.approx(0.9)
    assert meta.data(0.0253) == pytest.approx(0.1)


def test_extract_nb93():
    stub = StubEvaluation({
        (8, 4): NB93_MF8_MT4,
        (10, 4): NB93_MF10_MT4,
    })
    records = extract_isomeric_production(stub, source='ENDF/B-8.1')
    (record,) = records[4]
    assert record.mf == 10
    assert record.zap == 41093
    assert record.lfs == 1
    assert record.elfs == 30730.0
    assert record.qm == 0.0
    assert record.qi == -30730.0
    assert len(record.data.x) == 34
    assert record.data.x[0] == 31150.0
    # Partial cross section is kept in barns, never divided by a total
    assert record.data(14.356e6) == pytest.approx(3.756e-2)


def test_extract_skips_other_schemes():
    # An MF=8 subsection pointing at LMF=6 (scheme 2, e.g. TENDL MT=5)
    # must not produce a record even if an MF=10 section exists
    mf8 = NB93_MF8_MT4.replace(
        ' 4.109300+4 3.073000+4         10          1',
        ' 4.109300+4 3.073000+4          6          1')
    stub = StubEvaluation({(8, 4): mf8, (10, 4): NB93_MF10_MT4})
    assert extract_isomeric_production(stub, source='test') == {}


def test_assign_matched(am241_records):
    # ELIS matching: level 2 at 48630 eV matches the decay metastable at
    # 48600 eV, giving Am242_m1 (not the level-numbered Am242_m2)
    assignments = assign_levels(
        am241_records[102], 'Am242', {(95, 242): [(1, 48600.0)]}, 0.5)
    ground, meta = assignments
    assert ground.target == 'Am242'
    assert ground.status == GROUND
    assert meta.target == 'Am242_m1'
    assert meta.status == MATCHED
    assert meta.liso == 1
    assert meta.decay_elis == 48600.0


def test_assign_positional_fallback(am241_records):
    # An excitation energy failing the relative tolerance falls back to
    # positional pairing with the unused metastables
    assignments = assign_levels(
        am241_records[102], 'Am242', {(95, 242): [(1, 10000.0)]}, 0.5)
    meta = assignments[1]
    assert meta.target == 'Am242_m1'
    assert meta.status == POSITIONAL


def test_assign_folded(am241_records):
    # With no metastables in the decay data the level folds onto ground
    assignments = assign_levels(am241_records[102], 'Am242', {}, 0.5)
    meta = assignments[1]
    assert meta.target == 'Am242'
    assert meta.status == FOLDED


def test_assign_two_isomers_no_cross_match():
    # Ir192-like case: two levels must map to m1 and m2 without crossing
    records = extract_isomeric_production(StubEvaluation({
        (8, 102): AM241_MF8_MT102,
        (9, 102): AM241_MF9_MT102,
    }), source='test')[102]
    # Reuse the Am241 records but pretend there are two isomers, with the
    # level at 48630 eV clearly closer to the first
    isomers = {(95, 242): [(1, 48600.0), (2, 2200000.0)]}
    assignments = assign_levels(records, 'Am242', isomers, 0.5)
    assert assignments[1].target == 'Am242_m1'
    assert assignments[1].liso == 1


def test_group_records_merges_mf9_mf10():
    stub = StubEvaluation({
        (8, 4): NB93_MF8_MT4,
        (10, 4): NB93_MF10_MT4,
    })
    (record,) = extract_isomeric_production(stub, source='test')[4]
    duplicate = extract_isomeric_production(StubEvaluation({
        (8, 4): NB93_MF8_MT4,
        (10, 4): NB93_MF10_MT4,
    }), source='test')[4][0]
    duplicate.mf = 9
    grouped = group_records([
        (record, 'Nb93_m1'), (duplicate, 'Nb93_m1')])
    (production,) = grouped['Nb93_m1']
    assert production.level == 1
    assert [t.mf for t in production.tables] == [10, 9]


def test_scalar_ratios_thermal(am241_records):
    assignments = assign_levels(
        am241_records[102], 'Am242', {(95, 242): [(1, 48600.0)]}, 0.5)
    grouped = group_records(
        [(a.record, a.target) for a in assignments])
    ratios, note = compute_scalar_ratios(grouped, 'Am242', 'thermal')
    assert note is None
    assert ratios == {'Am242': pytest.approx(0.9),
                      'Am242_m1': pytest.approx(0.1)}


def test_scalar_ratios_flux_collapse(am241_records):
    assignments = assign_levels(
        am241_records[102], 'Am242', {(95, 242): [(1, 48600.0)]}, 0.5)
    grouped = group_records(
        [(a.record, a.target) for a in assignments])
    # All flux in a single low-energy group where the yield is constant
    mode = ([1.0e-5, 1.0, 3.0e7], [1.0, 0.0])
    ratios, note = compute_scalar_ratios(grouped, 'Am242', mode)
    assert note is None
    assert ratios['Am242_m1'] == pytest.approx(0.1)
    assert ratios['Am242'] == pytest.approx(0.9)


def test_scalar_ratios_mf10_fallback():
    stub = StubEvaluation({
        (8, 4): NB93_MF8_MT4,
        (10, 4): NB93_MF10_MT4,
    })
    (record,) = extract_isomeric_production(stub, source='test')[4]
    grouped = group_records([(record, 'Nb93_m1')])
    ratios, note = compute_scalar_ratios(grouped, 'Nb93', 'thermal')
    assert 'no MF=9 yield' in note
    assert ratios == {'Nb93': 1.0, 'Nb93_m1': 0.0}


def test_scalar_ratios_none_mode(am241_records):
    assignments = assign_levels(
        am241_records[102], 'Am242', {(95, 242): [(1, 48600.0)]}, 0.5)
    grouped = group_records(
        [(a.record, a.target) for a in assignments])
    ratios, note = compute_scalar_ratios(grouped, 'Am242', 'none')
    assert note is None
    assert ratios == {'Am242': 1.0, 'Am242_m1': 0.0}


def test_assign_positional_keeps_level_records_together():
    # One physical level carrying both an MF=9 and an MF=10 record must
    # land on a single target through the positional fallback, even when
    # more than one metastable is available
    records = extract_isomeric_production(StubEvaluation({
        (8, 102): AM241_MF8_MT102,
        (9, 102): AM241_MF9_MT102,
    }), source='test')[102]
    duplicate = extract_isomeric_production(StubEvaluation({
        (8, 102): AM241_MF8_MT102,
        (9, 102): AM241_MF9_MT102,
    }), source='test')[102][1]
    duplicate.mf = 10
    both = [records[1], duplicate]

    isomers = {(95, 242): [(1, 10000.0), (2, 2000000.0)]}
    assignments = assign_levels(both, 'Am242', isomers, 0.01)
    assert all(a.status == POSITIONAL for a in assignments)
    assert {a.target for a in assignments} == {'Am242_m1'}

    # With no metastables both records of the level fold together
    assignments = assign_levels(both, 'Am242', {}, 0.01)
    assert all(a.status == FOLDED for a in assignments)
    assert {a.target for a in assignments} == {'Am242'}


def test_select_records():
    primary = {102: ['p102'], 600: ['p600']}
    supplement = {102: ['s102']}

    # Primary wins over the supplement, first MT wins within a library,
    # and additionally populated MTs are reported back
    records, skipped = select_records({102, 600}, primary, supplement)
    assert records == ['p102']
    assert skipped == [600]

    # Supplement is used only when the primary has nothing for the MTs
    records, skipped = select_records({102}, {}, supplement)
    assert records == ['s102']
    assert skipped == []

    records, skipped = select_records({16}, primary, supplement)
    assert records == []
    assert skipped == []


def test_scalar_ratios_zero_bounded_group_structure(am241_records):
    # A first group starting at 0 eV has a geometric midpoint of 0, which
    # must be clamped into the MF=9 grid rather than silently dropping
    # the whole thermal group from the collapse
    assignments = assign_levels(
        am241_records[102], 'Am242', {(95, 242): [(1, 48600.0)]}, 0.5)
    grouped = group_records(
        [(a.record, a.target) for a in assignments])
    mode = ([0.0, 0.625, 2.0e7], [0.9, 0.1])
    ratios, note = compute_scalar_ratios(grouped, 'Am242', mode)
    assert note is None
    assert ratios['Am242_m1'] == pytest.approx(0.104007, rel=1e-3)
    assert ratios['Am242'] == pytest.approx(1.0 - ratios['Am242_m1'])


def test_report_write(tmp_path, am241_records):
    report = IsomerMappingReport(elis_rtol=0.5)
    assignments = assign_levels(
        am241_records[102], 'Am242', {(95, 242): [(1, 48600.0)]}, 0.5)
    for assignment in assignments:
        report.add('Am241', '(n,gamma)', assignment,
                   assignment.target, assignment.status)
    report.add_note('Am241', '(n,gamma)', 'example note')
    path = tmp_path / 'mapping.log'
    report.write(path)
    text = path.read_text()
    assert 'Am242_m1' in text
    assert 'matched: 1' in text
    assert 'ground: 1' in text
    assert 'example note' in text


def test_fold_when_target_missing_from_decay_data(am241_records):
    # A matched metastable whose nuclide is absent from the decay data is
    # folded onto the ground-state target via replace_missing
    class FakeHalfLife:
        nominal_value = 1000.0

    class FakeDecay:
        def __init__(self, stable):
            self.nuclide = {'stable': stable}
            self.half_life = FakeHalfLife()

    from openmc.deplete import Nuclide
    decay_data = {'Am241': FakeDecay(False), 'Am242': FakeDecay(False)}
    nuclide = Nuclide('Am241')
    report = IsomerMappingReport(elis_rtol=0.5)
    Chain._add_isomeric_reactions(
        nuclide, '(n,gamma)', 'Am242', 5537755.0, am241_records[102],
        decay_data, {(95, 242): [(1, 48600.0)]}, 0.5, 'none', report)

    targets = [rx.target for rx in nuclide.reactions]
    assert targets == ['Am242']
    productions = nuclide.isomeric_production[('(n,gamma)', 'Am242')]
    assert sorted(p.level for p in productions) == [0, 2]
    statuses = {row['status'] for row in report.rows}
    assert FOLDED in statuses


def test_from_endf_invalid_scalar_mode():
    with pytest.raises(ValueError, match='scalar_branching'):
        Chain.from_endf([], [], [], scalar_branching='banana')


def test_from_endf_scalar_tuple_validation():
    with pytest.raises(ValueError, match='one longer'):
        Chain.from_endf([], [], [], isomeric_branching=True,
                        scalar_branching=([1.0, 2.0], [1.0, 2.0]))
    with pytest.raises(ValueError, match='positive sum'):
        Chain.from_endf([], [], [], isomeric_branching=True,
                        scalar_branching=([1.0, 2.0], [0.0]))
    with pytest.raises(ValueError, match='energies, flux'):
        Chain.from_endf([], [], [], isomeric_branching=True,
                        scalar_branching=(1.0, 2.0, 3.0))


def test_from_endf_kwargs_require_flag():
    with pytest.raises(ValueError, match='isomeric_branching=True'):
        Chain.from_endf([], [], [], branching_files=['some_file'])
    with pytest.raises(ValueError, match='isomeric_branching=True'):
        Chain.from_endf([], [], [], isomer_mapping_log='log.txt')
    with pytest.raises(ValueError, match='isomeric_branching=True'):
        Chain.from_endf([], [], [], scalar_branching='thermal')


def _find_endf(directory, patterns):
    for pattern in patterns:
        matches = sorted(Path(directory).glob(pattern))
        if matches:
            return matches[0]
    pytest.skip(f'No file matching {patterns} under {directory}')


@pytest.mark.skipif(
    'OPENMC_ENDF_DATA' not in os.environ,
    reason='OPENMC_ENDF_DATA environment variable must be set')
def test_from_endf_isomeric():
    endf_data = Path(os.environ['OPENMC_ENDF_DATA'])
    neutron = _find_endf(endf_data / 'neutrons', ['*Am*241*'])
    decay_files = sorted((endf_data / 'decay').glob('*.endf'))
    fpy = _find_endf(endf_data / 'nfy', ['*U*235*'])

    chain = Chain.from_endf(
        decay_files, [fpy], [neutron],
        reactions=['(n,gamma)'],
        progress=False,
        isomeric_branching=True,
        scalar_branching='thermal')

    am241 = chain['Am241']
    targets = {rx.target: rx.branching_ratio for rx in am241.reactions
               if rx.type == '(n,gamma)'}
    assert targets == {'Am242': pytest.approx(0.9),
                       'Am242_m1': pytest.approx(0.1)}
    production = am241.isomeric_production[('(n,gamma)', 'Am242_m1')]
    assert production[0].level == 2
    assert production[0].tables[0].mf == 9
    assert production[0].tables[0].data(0.0253) == pytest.approx(0.1)
    assert chain.validate(strict=True)


@pytest.mark.skipif(
    'OPENMC_ENDF_DATA' not in os.environ,
    reason='OPENMC_ENDF_DATA environment variable must be set')
def test_from_endf_isomeric_no_data_identical(tmp_path):
    # With isomeric_branching=True but no MF=8/9/10 data for the
    # requested reaction, the chain must be identical to the default one
    endf_data = Path(os.environ['OPENMC_ENDF_DATA'])
    neutron = _find_endf(endf_data / 'neutrons', ['*Am*241*'])
    decay_files = sorted((endf_data / 'decay').glob('*.endf'))
    fpy = _find_endf(endf_data / 'nfy', ['*U*235*'])

    # Am241 carries MF=9 only for (n,gamma); (n,2n) has no isomeric data
    common = dict(decay_files=decay_files, fpy_files=[fpy],
                  neutron_files=[neutron], reactions=['(n,2n)'],
                  progress=False)
    chain_default = Chain.from_endf(**common)
    chain_isomeric = Chain.from_endf(**common, isomeric_branching=True)

    default_xml = tmp_path / 'default.xml'
    isomeric_xml = tmp_path / 'isomeric.xml'
    chain_default.export_to_xml(default_xml)
    chain_isomeric.export_to_xml(isomeric_xml)
    assert default_xml.read_text() == isomeric_xml.read_text()
