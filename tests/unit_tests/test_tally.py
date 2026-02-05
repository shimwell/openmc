import openmc


def test_tally_init_args():
    """Test that Tally constructor kwargs are applied correctly."""
    f = openmc.EnergyFilter([0.0, 1.0, 20.0e6])
    t = openmc.Tally(
        name='my tally',
        scores=['flux', 'fission'],
        filters=[f],
        nuclides=['U235'],
        estimator='tracklength',
    )

    assert t.name == 'my tally'
    assert t.scores == ['flux', 'fission']
    assert t.filters == [f]
    assert t.nuclides == ['U235']
    assert t.estimator == 'tracklength'
