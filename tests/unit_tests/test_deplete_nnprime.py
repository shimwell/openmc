"""Tests for (n,n') inelastic isomeric activation in depletion chains."""

import os
from pathlib import Path

import numpy as np
import pytest

import openmc.data
from openmc.deplete import Chain, reaction_rates
from openmc.deplete.chain import REACTIONS

NNPRIME_CHAIN = """<depletion_chain>
  <nuclide name="A" reactions="2">
    <reaction type="(n,n')" Q="0.0" target="A" branching_ratio="0.9"/>
    <reaction type="(n,n')" Q="0.0" target="A_m1" branching_ratio="0.1"/>
  </nuclide>
  <nuclide name="A_m1" reactions="1">
    <reaction type="(n,n')" Q="0.0" target="A" branching_ratio="1.0"/>
  </nuclide>
</depletion_chain>
"""


def test_reaction_registry():
    assert REACTIONS["(n,n')"].mts == {4}
    assert REACTIONS["(n,n')"].secondaries == ()
    assert openmc.data.DADZ["(n,n')"] == (0, 0)
    assert openmc.data.REACTION_MT["(n,n')"] == 4


def test_self_loop_matrix_conserves_atoms(tmp_path):
    chain_file = tmp_path / "chain.xml"
    chain_file.write_text(NNPRIME_CHAIN)
    chain = Chain.from_xml(chain_file)
    assert chain.reactions == ["(n,n')"]

    rates = reaction_rates.ReactionRates(["1"], ["A", "A_m1"], ["(n,n')"])
    rates.set("1", "A", "(n,n')", 2.0)
    rates.set("1", "A_m1", "(n,n')", 0.5)
    matrix = chain.form_matrix(rates[0]).toarray()

    # A loses at the full rate but 90% returns as a self-loop; A_m1 fully
    # de-excites back to ground
    assert matrix[0, 0] == pytest.approx(-2.0 + 2.0 * 0.9)
    assert matrix[1, 0] == pytest.approx(2.0 * 0.1)
    assert matrix[0, 1] == pytest.approx(0.5)
    assert matrix[1, 1] == pytest.approx(-0.5)

    # Atom conservation: every column sums to zero
    assert np.allclose(matrix.sum(axis=0), 0.0)


def _find_endf(directory, patterns):
    for pattern in patterns:
        matches = sorted(Path(directory).glob(pattern))
        if matches:
            return matches[0]
    pytest.skip(f'No file matching {patterns} under {directory}')


@pytest.mark.skipif(
    'OPENMC_ENDF_DATA' not in os.environ,
    reason='OPENMC_ENDF_DATA environment variable must be set')
def test_from_endf_nnprime():
    endf_data = Path(os.environ['OPENMC_ENDF_DATA'])
    decay_files = sorted((endf_data / 'decay').glob('*.endf'))
    fpy = _find_endf(endf_data / 'nfy', ['*U*235*'])
    nb93 = _find_endf(endf_data / 'neutrons', ['*Nb*93*'])
    am241 = _find_endf(endf_data / 'neutrons', ['*Am*241*'])

    chain = Chain.from_endf(
        decay_files, [fpy], [nb93, am241],
        reactions=["(n,n')"],
        progress=False,
        isomeric_branching=True)

    # Nb93 has MF=10 data for MT=4: a ground self-loop plus the m1 target
    nb = chain['Nb93']
    targets = {rx.target: rx.branching_ratio for rx in nb.reactions
               if rx.type == "(n,n')"}
    assert targets == {'Nb93': 1.0, 'Nb93_m1': 0.0}
    production = nb.isomeric_production[("(n,n')", 'Nb93_m1')]
    assert production[0].level == 1
    table = production[0].tables[0]
    assert table.mf == 10
    assert len(table.data.x) == 34
    assert table.data(14.356e6) == pytest.approx(3.756e-2)

    # Am241 has no MF=8/9/10 for MT=4, so no self-loop entries are added
    assert all(rx.type != "(n,n')" for rx in chain['Am241'].reactions)

    assert chain.validate(strict=True)
