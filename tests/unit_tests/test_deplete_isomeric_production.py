"""Tests for energy-dependent isomeric production data in depletion chains."""

import lxml.etree as ET
import numpy as np
import pytest

from openmc.data import Tabulated1D
from openmc.deplete import Chain, IsomericProduction, Nuclide, ProductionTable
from openmc.deplete.nuclide import ReactionTuple

# Values from the ENDF/B-VIII.1 Am241 evaluation, MF=9, MT=102
AM241_ENERGIES = ("1e-05 0.369 1000.0 100000.0 600000.1 1000000.0 "
                  "2000000.0 4000000.1 30000000.0")
AM241_GROUND = "0.9 0.9 0.8667 0.842 0.81533 0.74382 0.5703 0.52 0.52"
AM241_M1 = "0.1 0.1 0.1333 0.158 0.18467 0.25618 0.4297 0.48 0.48"

CHAIN_XML = f"""<depletion_chain>
  <nuclide name="Am241" half_life="13651840000.0" decay_modes="1" decay_energy="5627905.0" reactions="2">
    <decay type="alpha" target="Nothing" branching_ratio="1.0"/>
    <reaction type="(n,gamma)" Q="5537755.0" target="Am242">
      <isomeric_production level="0" excitation_energy="0.0">
        <table mf="9" mt="102" source="ENDF/B-VIII.1" QM="5537755.0" QI="5537755.0" breakpoints="9" interpolation="3">
          <energies>{AM241_ENERGIES}</energies>
          <values>{AM241_GROUND}</values>
        </table>
      </isomeric_production>
    </reaction>
    <reaction type="(n,gamma)" Q="5537755.0" target="Am242_m1" branching_ratio="0.0">
      <isomeric_production level="2" excitation_energy="48630.0">
        <table mf="9" mt="102" source="ENDF/B-VIII.1" QM="5537755.0" QI="5489125.0" breakpoints="9" interpolation="3">
          <energies>{AM241_ENERGIES}</energies>
          <values>{AM241_M1}</values>
        </table>
      </isomeric_production>
    </reaction>
  </nuclide>
  <nuclide name="Nb93" reactions="2">
    <reaction type="(n,gamma)" Q="7229000.0" target="Nb94">
      <isomeric_production level="0" excitation_energy="0.0">
        <table mf="9" mt="102" source="TENDL-2025" QM="7229000.0" QI="7229000.0" breakpoints="2 4" interpolation="2 5">
          <energies>1e-05 1000.0 100000.0 20000000.0</energies>
          <values>0.5 0.55 0.6 0.7</values>
        </table>
        <table mf="10" mt="102" source="TENDL-2025" QM="7229000.0" QI="7229000.0" breakpoints="4" interpolation="2">
          <energies>1e-05 1000.0 100000.0 20000000.0</energies>
          <values>0.5 0.02 0.001 1e-05</values>
        </table>
      </isomeric_production>
      <isomeric_production level="7" excitation_energy="1300000.0">
        <table mf="10" mt="102" source="TENDL-2025" QM="7229000.0" QI="5929000.0" breakpoints="3" interpolation="2">
          <energies>1500000.0 5000000.0 20000000.0</energies>
          <values>0.0 0.001 1e-05</values>
        </table>
      </isomeric_production>
    </reaction>
    <reaction type="(n,gamma)" Q="7229000.0" target="Nb94_m1" branching_ratio="0.0">
      <isomeric_production level="1" excitation_energy="40902.0">
        <table mf="9" mt="102" source="TENDL-2025" QM="7229000.0" QI="7188098.0" breakpoints="4" interpolation="2">
          <energies>1e-05 1000.0 100000.0 20000000.0</energies>
          <values>0.5 0.45 0.4 0.3</values>
        </table>
      </isomeric_production>
    </reaction>
  </nuclide>
  <nuclide name="Am242" reactions="0"/>
  <nuclide name="Am242_m1" reactions="0"/>
  <nuclide name="Nb94" reactions="0"/>
  <nuclide name="Nb94_m1" reactions="0"/>
</depletion_chain>
"""


@pytest.fixture
def chain(tmp_path):
    chain_file = tmp_path / "chain.xml"
    chain_file.write_text(CHAIN_XML)
    return Chain.from_xml(chain_file)


def test_from_xml(chain):
    am241 = chain["Am241"]

    # Reaction tuples are unchanged by the new child elements
    assert am241.reactions == [
        ReactionTuple("(n,gamma)", "Am242", 5537755.0, 1.0),
        ReactionTuple("(n,gamma)", "Am242_m1", 5537755.0, 0.0),
    ]

    # Ground and metastable entries each carry their own verbatim table
    ground = am241.isomeric_production[("(n,gamma)", "Am242")]
    meta = am241.isomeric_production[("(n,gamma)", "Am242_m1")]
    assert len(ground) == 1 and len(meta) == 1
    assert ground[0].level == 0
    assert ground[0].excitation_energy == 0.0
    assert meta[0].level == 2
    assert meta[0].excitation_energy == 48630.0

    table = meta[0].tables[0]
    assert table.mf == 9
    assert table.mt == 102
    assert table.source == "ENDF/B-VIII.1"
    assert table.QM == 5537755.0
    assert table.QI == 5489125.0
    np.testing.assert_array_equal(table.data.breakpoints, [9])
    np.testing.assert_array_equal(table.data.interpolation, [3])
    np.testing.assert_array_equal(
        table.data.x, [float(x) for x in AM241_ENERGIES.split()])
    np.testing.assert_array_equal(
        table.data.y, [float(y) for y in AM241_M1.split()])

    # The tables are callable functions of energy
    assert table.data(0.0253) == pytest.approx(0.1)
    assert ground[0].tables[0].data(0.0253) == pytest.approx(0.9)


def test_multiple_tables_and_folded_levels(chain):
    nb93 = chain["Nb93"]
    ground = nb93.isomeric_production[("(n,gamma)", "Nb94")]

    # Ground target holds its own level plus a folded unmappable level
    assert [p.level for p in ground] == [0, 7]

    # MF=9 and MF=10 tables coexist on one level
    assert [t.mf for t in ground[0].tables] == [9, 10]

    # Multi-region interpolation survives parsing
    table = ground[0].tables[0]
    np.testing.assert_array_equal(table.data.breakpoints, [2, 4])
    np.testing.assert_array_equal(table.data.interpolation, [2, 5])

    # The folded level keeps its own excitation energy and threshold grid
    folded = ground[1]
    assert folded.excitation_energy == 1300000.0
    assert folded.tables[0].data.x[0] == 1500000.0


def test_xml_roundtrip(chain, tmp_path):
    out = tmp_path / "chain_out.xml"
    chain.export_to_xml(out)
    reread = Chain.from_xml(out)

    for nuclide in chain.nuclides:
        other = reread[nuclide.name]
        assert other.reactions == nuclide.reactions
        assert other.isomeric_production == nuclide.isomeric_production

    # A second write produces identical bytes (bit-exact round-trip)
    out2 = tmp_path / "chain_out2.xml"
    reread.export_to_xml(out2)
    assert out.read_text() == out2.read_text()


def test_legacy_reader_compatibility(chain, tmp_path):
    # A reader that only looks at <reaction> attributes sees a normal chain
    out = tmp_path / "chain_out.xml"
    chain.export_to_xml(out)
    root = ET.parse(str(out))
    am241 = root.find('nuclide[@name="Am241"]')
    reactions = am241.findall("reaction")
    assert len(reactions) == 2
    assert reactions[0].get("target") == "Am242"
    assert reactions[0].get("branching_ratio") is None
    assert reactions[1].get("target") == "Am242_m1"
    assert reactions[1].get("branching_ratio") == "0.0"

    # Scalar branching ratios per reaction type still sum to one
    assert chain.validate(strict=True)


def test_validate_orphan_entry():
    nuc = Nuclide("A")
    nuc.add_reaction("(n,gamma)", "B", 0.0, 1.0)
    table = ProductionTable(
        mf=9, mt=102, source="test", QM=0.0, QI=0.0,
        data=Tabulated1D([1.0, 2.0], [0.5, 0.5]))
    nuc.isomeric_production[("(n,gamma)", "B_m1")] = [
        IsomericProduction(1, 100.0, [table])]
    with pytest.raises(ValueError, match="not a reaction present"):
        nuc.validate(strict=True)
    with pytest.warns(UserWarning, match="not a reaction present"):
        assert not nuc.validate(strict=False)


def test_validate_mf9_range():
    nuc = Nuclide("A")
    nuc.add_reaction("(n,gamma)", "B", 0.0, 1.0)
    table = ProductionTable(
        mf=9, mt=102, source="test", QM=0.0, QI=0.0,
        data=Tabulated1D([1.0, 2.0], [0.5, 1.2]))
    nuc.isomeric_production[("(n,gamma)", "B")] = [
        IsomericProduction(0, 0.0, [table])]
    with pytest.raises(ValueError, match="outside of"):
        nuc.validate(strict=True)


def test_validate_mf10_negative():
    nuc = Nuclide("A")
    nuc.add_reaction("(n,gamma)", "B", 0.0, 1.0)
    table = ProductionTable(
        mf=10, mt=102, source="test", QM=0.0, QI=0.0,
        data=Tabulated1D([1.0, 2.0], [0.1, -0.1]))
    nuc.isomeric_production[("(n,gamma)", "B")] = [
        IsomericProduction(0, 0.0, [table])]
    with pytest.raises(ValueError, match="negative MF=10"):
        nuc.validate(strict=True)


def test_validate_mf9_sum():
    # Ground and metastable MF=9 yields that sum above one must fail
    nuc = Nuclide("A")
    nuc.add_reaction("(n,gamma)", "B", 0.0, 1.0)
    nuc.add_reaction("(n,gamma)", "B_m1", 0.0, 0.0)
    ground = ProductionTable(
        mf=9, mt=102, source="test", QM=0.0, QI=0.0,
        data=Tabulated1D([1.0, 2.0], [0.9, 0.9]))
    meta = ProductionTable(
        mf=9, mt=102, source="test", QM=0.0, QI=-100.0,
        data=Tabulated1D([1.0, 2.0], [0.15, 0.1]))
    nuc.isomeric_production[("(n,gamma)", "B")] = [
        IsomericProduction(0, 0.0, [ground])]
    nuc.isomeric_production[("(n,gamma)", "B_m1")] = [
        IsomericProduction(1, 100.0, [meta])]
    with pytest.raises(ValueError, match="sum to"):
        nuc.validate(strict=True)


def test_validate_implicit_ground_share():
    # A metastable-only MF=9 yield (like In115 capture, Y=0.79 to In116_m1)
    # has an implicit ground share, so no sum rule applies
    nuc = Nuclide("In115")
    nuc.add_reaction("(n,gamma)", "In116", 0.0, 1.0)
    nuc.add_reaction("(n,gamma)", "In116_m1", 0.0, 0.0)
    table = ProductionTable(
        mf=9, mt=102, source="test", QM=6784719.0, QI=6657319.0,
        data=Tabulated1D([1e-5, 3e7], [0.79, 0.79]))
    nuc.isomeric_production[("(n,gamma)", "In116_m1")] = [
        IsomericProduction(2, 127400.0, [table])]
    assert nuc.validate(strict=True)


def test_validate_quiet():
    nuc = Nuclide("A")
    nuc.add_reaction("(n,gamma)", "B", 0.0, 1.0)
    table = ProductionTable(
        mf=9, mt=102, source="test", QM=0.0, QI=0.0,
        data=Tabulated1D([1.0, 2.0], [0.5, 1.2]))
    nuc.isomeric_production[("(n,gamma)", "B")] = [
        IsomericProduction(0, 0.0, [table])]
    assert not nuc.validate(strict=False, quiet=True)


def test_no_isomeric_data():
    nuc = Nuclide("A")
    nuc.add_reaction("(n,gamma)", "B", 0.0, 1.0)
    assert nuc.isomeric_production == {}
    assert nuc.validate(strict=True)
    elem = nuc.to_xml_element()
    assert elem.find("reaction").find("isomeric_production") is None


def test_duplicate_reaction_pair_roundtrip(tmp_path):
    # Production data attached to a (type, target) pair that appears on
    # more than one reaction element must not multiply on round trip
    nuc = Nuclide("A")
    nuc.add_reaction("(n,gamma)", "B", 0.0, 0.5)
    nuc.add_reaction("(n,gamma)", "B", 0.0, 0.5)
    table = ProductionTable(
        mf=9, mt=102, source="test", QM=0.0, QI=0.0,
        data=Tabulated1D([1.0, 2.0], [0.5, 0.5]))
    nuc.isomeric_production[("(n,gamma)", "B")] = [
        IsomericProduction(0, 0.0, [table])]

    elem = nuc.to_xml_element()
    assert len(elem.findall("reaction/isomeric_production")) == 1
    reread = Nuclide.from_xml(elem)
    assert len(reread.isomeric_production[("(n,gamma)", "B")]) == 1


def test_orphan_entry_not_written():
    nuc = Nuclide("A")
    nuc.add_reaction("(n,gamma)", "B", 0.0, 1.0)
    table = ProductionTable(
        mf=9, mt=102, source="test", QM=0.0, QI=0.0,
        data=Tabulated1D([1.0, 2.0], [0.5, 0.5]))
    nuc.isomeric_production[("(n,gamma)", "B_m1")] = [
        IsomericProduction(1, 100.0, [table])]
    elem = nuc.to_xml_element()
    assert elem.findall("reaction/isomeric_production") == []


def test_validate_reaction_sum_message():
    # A nuclide with no decay modes and inconsistent reaction sums must
    # report the reaction sum (this previously raised a NameError)
    nuc = Nuclide("A")
    nuc.add_reaction("(n,gamma)", "B", 0.0, 0.5)
    nuc.add_reaction("(n,gamma)", "B_m1", 0.0, 0.25)
    with pytest.raises(ValueError, match="sum to 0.75"):
        nuc.validate(strict=True)


def test_get_isomeric_production(chain):
    data = chain.get_isomeric_production("Am241", "(n,gamma)")
    assert set(data) == {"Am242", "Am242_m1"}
    assert data["Am242_m1"][0].level == 2
    assert chain.get_isomeric_production("Am242", "(n,gamma)") == {}


def test_reduce_carries_data(chain):
    reduced = chain.reduce(["Am241"])
    original = chain["Am241"].isomeric_production
    carried = reduced["Am241"].isomeric_production
    assert carried == original
    # Deep copy, not shared references
    key = ("(n,gamma)", "Am242_m1")
    assert carried[key][0] is not original[key][0]


def test_reduce_drops_data_with_warning(chain):
    with pytest.warns(UserWarning, match="dropped"):
        reduced = chain.reduce(["Nb93"], level=0)
    nb93 = reduced["Nb93"]
    assert nb93.isomeric_production == {}
    # Total destruction rate entries are still present, with no target
    assert all(rx.target is None for rx in nb93.reactions)


def test_set_branch_ratios_preserves_data(chain):
    before = dict(chain["Am241"].isomeric_production)
    chain.set_branch_ratios({"Am241": {"Am242": 0.89, "Am242_m1": 0.11}})
    am241 = chain["Am241"]
    ratios = {rx.target: rx.branching_ratio for rx in am241.reactions
              if rx.type == "(n,gamma)"}
    assert ratios == {"Am242": 0.89, "Am242_m1": 0.11}
    assert am241.isomeric_production == before


def test_set_branch_ratios_infers_ground_and_preserves(chain):
    before = dict(chain["Am241"].isomeric_production)
    chain.set_branch_ratios({"Am241": {"Am242_m1": 0.1}})
    am241 = chain["Am241"]
    ratios = {rx.target: rx.branching_ratio for rx in am241.reactions
              if rx.type == "(n,gamma)"}
    assert ratios == {"Am242": pytest.approx(0.9), "Am242_m1": 0.1}
    assert am241.isomeric_production == before


def test_set_branch_ratios_discard_protection(chain):
    # Removing a target that carries data raises by default
    with pytest.raises(ValueError, match="Am241 -> Am242_m1"):
        chain.set_branch_ratios({"Am241": {"Am242": 1.0}})

    # Nothing was mutated by the failed call
    assert ("(n,gamma)", "Am242_m1") in chain["Am241"].isomeric_production

    # Opting out drops the data with a warning
    with pytest.warns(UserWarning, match="Am241 -> Am242_m1"):
        chain.set_branch_ratios({"Am241": {"Am242": 1.0}},
                                preserve_isomeric_data=False)
    am241 = chain["Am241"]
    assert ("(n,gamma)", "Am242_m1") not in am241.isomeric_production
    targets = [rx.target for rx in am241.reactions if rx.type == "(n,gamma)"]
    assert targets == ["Am242"]
    # The surviving target keeps its data
    assert ("(n,gamma)", "Am242") in am241.isomeric_production
