from pathlib import Path
from math import exp

import numpy as np
import pytest
import openmc
import openmc.deplete
from openmc.deplete import d1s


CHAIN_PATH = Path(__file__).parents[1] / "chain_ni.xml"


@pytest.fixture
def model():
    """Simple model with natural Ni"""
    mat = openmc.Material()
    mat.add_element('Ni', 1.0)
    geom = openmc.Geometry([openmc.Cell(fill=mat)])
    return openmc.Model(geometry=geom)


def test_get_radionuclides(model):
    # Check that radionuclides are correct and are unstable
    chain = openmc.deplete.Chain.from_xml(CHAIN_PATH)
    nuclides = d1s.get_radionuclides(model, chain)
    assert sorted(nuclides) == [
        'Co58', 'Co60', 'Co61', 'Co62', 'Co64',
        'Fe55', 'Fe59', 'Fe61', 'Ni57', 'Ni59', 'Ni63', 'Ni65'
    ]
    for nuc in nuclides:
        assert openmc.data.half_life(nuc) is not None


@pytest.mark.parametrize("nuclide", ['Co60', 'Ni63', 'H3', 'Na24', 'K40'])
def test_time_correction_factors(nuclide):
    # Irradiation schedule turning unit neutron source on and off
    timesteps = [1.0, 1.0, 1.0]
    source_rates = [1.0, 0.0, 1.0]

    # Compute expected solution
    decay_rate = openmc.data.decay_constant(nuclide)
    g = exp(-decay_rate)
    expected = [0.0, (1 - g), (1 - g)*g, (1 - g)*(1 + g*g)]

    # Test against expected solution
    tcf = d1s.time_correction_factors([nuclide], timesteps, source_rates)
    assert tcf[nuclide] == pytest.approx(expected)

    # Make sure all values at first timestep and onward are positive (K40 case
    # has very small decay constant that stresses this)
    assert np.all(tcf[nuclide][1:] > 0.0)

    # Timesteps as a tuple
    timesteps = [(1.0, 's'), (1.0, 's'), (1.0, 's')]
    tcf = d1s.time_correction_factors([nuclide], timesteps, source_rates)
    assert tcf[nuclide] == pytest.approx(expected)

    # Test changing units
    timesteps = [1.0/60.0, 1.0/60.0, 1.0/60.0]
    tcf = d1s.time_correction_factors([nuclide], timesteps, source_rates,
                                      timestep_units='min')
    assert tcf[nuclide] == pytest.approx(expected)


def test_prepare_tallies(model):
    tally = openmc.Tally()
    tally.filters = [openmc.ParticleFilter('photon')]
    tally.scores = ['flux']
    model.tallies = [tally]

    # Check that prepare_tallies adds a ParentNuclideFilter
    nuclides = ['Co58', 'Co60', 'Fe55']
    d1s.prepare_tallies(model, nuclides, chain_file=CHAIN_PATH)
    assert tally.contains_filter(openmc.ParentNuclideFilter)
    assert list(tally.filters[-1].bins) == nuclides

    # Get rid of parent nuclide filter
    tally.filters.pop()

    # With no nuclides specified, filter should use get_radionuclides
    radionuclides = d1s.get_radionuclides(model, CHAIN_PATH)
    d1s.prepare_tallies(model, chain_file=CHAIN_PATH)
    assert tally.contains_filter(openmc.ParentNuclideFilter)
    assert sorted(tally.filters[-1].bins) == sorted(radionuclides)

    assert len(tally.filters) == 2
    # calling prepare_tallies twice should not add another ParentNuclideFilter
    d1s.prepare_tallies(model, chain_file=CHAIN_PATH)
    assert len(tally.filters) == 2


def test_apply_time_correction(run_in_tmpdir):
    # Make simple sphere model with elemental Ni
    mat = openmc.Material()
    mat.add_element('Ni', 1.0)
    sphere = openmc.Sphere(r=10.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-sphere)
    model = openmc.Model()
    model.geometry = openmc.Geometry([cell])
    model.settings.run_mode = 'fixed source'
    model.settings.batches = 3
    model.settings.particles = 10
    model.settings.photon_transport = True
    model.settings.use_decay_photons = True
    particle_filter = openmc.ParticleFilter('photon')
    tally = openmc.Tally()
    tally.filters = [particle_filter]
    tally.scores = ['flux']
    model.tallies = [tally]

    # Prepare tallies for D1S and compute time correction factors
    nuclides = d1s.prepare_tallies(model, chain_file=CHAIN_PATH)
    factors = d1s.time_correction_factors(nuclides, [1.0e10], [1.0])

    # Run OpenMC and get tally result
    with openmc.config.patch('chain_file', CHAIN_PATH):
        output_path = model.run()
    with openmc.StatePoint(output_path) as sp:
        tally = sp.tallies[tally.id]
        flux = tally.mean.flatten()

    # Copy attributes from original tally
    tally_filters = list(tally.filters)
    tally_sum = tally.sum.copy()
    tally_sum_sq = tally.sum_sq.copy()
    tally_mean = tally.mean.copy()
    tally_std_dev = tally.std_dev.copy()

    # Apply TCF and make sure results are consistent
    result = d1s.apply_time_correction(tally, factors, sum_nuclides=False)
    tcf = np.array([factors[nuc][-1] for nuc in nuclides])
    assert result.mean.flatten() == pytest.approx(tcf * flux)

    # Make sure summed results match a manual sum
    result_summed = d1s.apply_time_correction(tally, factors)
    assert result_summed.mean.flatten()[0] == pytest.approx(result.mean.sum())

    # Make sure original tally is unchanged
    assert tally.filters == tally_filters
    assert np.all(tally.sum == tally_sum)
    assert np.all(tally.sum_sq == tally_sum_sq)
    assert np.all(tally.mean == tally_mean)
    assert np.all(tally.std_dev == tally_std_dev)

    # Make sure various tally methods work
    result.get_values()
    result_summed.get_values()
    result.get_reshaped_data()
    result_summed.get_reshaped_data()
    result.get_pandas_dataframe()
    result_summed.get_pandas_dataframe()


def test_apply_time_correction_series(run_in_tmpdir):
    # Build the same model used in test_apply_time_correction
    mat = openmc.Material()
    mat.add_element('Ni', 1.0)
    sphere = openmc.Sphere(r=10.0, boundary_type='vacuum')
    cell = openmc.Cell(fill=mat, region=-sphere)
    model = openmc.Model()
    model.geometry = openmc.Geometry([cell])
    model.settings.run_mode = 'fixed source'
    model.settings.batches = 3
    model.settings.particles = 10
    model.settings.photon_transport = True
    model.settings.use_decay_photons = True
    particle_filter = openmc.ParticleFilter('photon')
    tally = openmc.Tally()
    tally.filters = [particle_filter]
    tally.scores = ['flux']
    model.tallies = [tally]

    # A schedule with several timesteps so the series has > 1 entry.
    nuclides = d1s.prepare_tallies(model, chain_file=CHAIN_PATH)
    timesteps = [1.0e8, 1.0e8, 1.0e8, 1.0e8]
    source_rates = [1.0, 0.0, 1.0, 0.0]
    factors = d1s.time_correction_factors(nuclides, timesteps, source_rates)
    n_times = len(factors[nuclides[0]])

    # Run the model once
    with openmc.config.patch('chain_file', CHAIN_PATH):
        output_path = model.run()
    with openmc.StatePoint(output_path) as sp:
        tally = sp.tallies[tally.id]

        # Snapshot original tally state so we can confirm immutability later
        orig_filters = list(tally.filters)
        orig_sum = tally.sum.copy()
        orig_sum_sq = tally.sum_sq.copy()
        orig_mean = tally.mean.copy()
        orig_std_dev = tally.std_dev.copy()

        # sum_nuclides=True: series matches per-index loop
        mean_series, std_series = d1s.apply_time_correction_series(
            tally, factors, sum_nuclides=True
        )
        assert mean_series.shape[0] == n_times
        assert std_series.shape == mean_series.shape

        for i in range(n_times):
            ref = d1s.apply_time_correction(
                tally, factors, index=i, sum_nuclides=True
            )
            np.testing.assert_allclose(
                mean_series[i].reshape(ref.mean.shape), ref.mean
            )
            np.testing.assert_allclose(
                std_series[i].reshape(ref.std_dev.shape), ref.std_dev
            )

        # sum_nuclides=False: series matches per-index loop
        mean_series_f, std_series_f = d1s.apply_time_correction_series(
            tally, factors, sum_nuclides=False
        )
        assert mean_series_f.shape[0] == n_times

        for i in range(n_times):
            ref = d1s.apply_time_correction(
                tally, factors, index=i, sum_nuclides=False
            )
            np.testing.assert_allclose(
                mean_series_f[i].reshape(ref.mean.shape), ref.mean
            )
            np.testing.assert_allclose(
                std_series_f[i].reshape(ref.std_dev.shape), ref.std_dev
            )

        # explicit indices subset (and unordered)
        subset = [n_times - 1, 0, 2]
        mean_sub, std_sub = d1s.apply_time_correction_series(
            tally, factors, indices=subset
        )
        assert mean_sub.shape[0] == len(subset)
        for k, i in enumerate(subset):
            ref = d1s.apply_time_correction(tally, factors, index=i)
            np.testing.assert_allclose(
                mean_sub[k].reshape(ref.mean.shape), ref.mean
            )
            np.testing.assert_allclose(
                std_sub[k].reshape(ref.std_dev.shape), ref.std_dev
            )

        # original tally is unchanged
        assert tally.filters == orig_filters
        assert np.all(tally.sum == orig_sum)
        assert np.all(tally.sum_sq == orig_sum_sq)
        assert np.all(tally.mean == orig_mean)
        assert np.all(tally.std_dev == orig_std_dev)

        # missing ParentNuclideFilter raises
        bare = openmc.Tally()
        bare.filters = [particle_filter]
        bare.scores = ['flux']
        with pytest.raises(ValueError):
            d1s.apply_time_correction_series(bare, factors)
