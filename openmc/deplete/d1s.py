"""D1S module

This module contains functionality to support the direct 1-step (D1S) method for
shutdown dose rate calculations.

"""

from copy import copy
from typing import Sequence
from math import log, prod

import numpy as np

import openmc
from openmc.data import half_life
from .abc import _normalize_timesteps
from .chain import Chain, _get_chain
from ..checkvalue import PathLike


def get_radionuclides(model: openmc.Model, chain_file: PathLike | Chain | None = None) -> list[str]:
    """Determine all radionuclides that can be produced during D1S.

    Parameters
    ----------
    model : openmc.Model
        Model that should be used for determining what nuclides are present
    chain_file : PathLike | Chain
        Path to the depletion chain XML file or instance of openmc.deplete.Chain.
        Used for inspecting decay data. Defaults to ``openmc.config['chain_file']``

    Returns
    -------
    List of nuclide names

    """

    # Determine what nuclides appear in the model
    model_nuclides = {nuc for mat in model._materials_by_id.values()
                      for nuc in mat.get_nuclides()}

    # Load chain file
    chain = _get_chain(chain_file)

    radionuclides = set()
    for nuclide in chain.nuclides:
        # Restrict to set of nuclides present in model
        if nuclide.name not in model_nuclides:
            continue

        # Loop over reactions and add any targets that are unstable
        for rx_tuple in nuclide.reactions:
            target = rx_tuple.target
            if target is None:
                continue
            target_nuclide = chain[target]
            if target_nuclide.half_life is not None:
                radionuclides.add(target_nuclide.name)

    return list(radionuclides)


def time_correction_factors(
        nuclides: list[str],
        timesteps: Sequence[float] | Sequence[tuple[float, str]],
        source_rates: float | Sequence[float],
        timestep_units: str = 's'
) -> dict[str, np.ndarray]:
    """Calculate time correction factors for the D1S method.

    This function determines the time correction factor that should be applied
    to photon tallies as part of the D1S method.

    Parameters
    ----------
    nuclides : list of str
        The name of the nuclide to find the time correction for, e.g., 'Ni65'
    timesteps : iterable of float or iterable of tuple
        Array of timesteps. Note that values are not cumulative. The units are
        specified by the `timestep_units` argument when `timesteps` is an
        iterable of float. Alternatively, units can be specified for each step
        by passing a sequence of (value, unit) tuples.
    source_rates : float or iterable of float
        Source rate in [neutron/sec] for each interval in `timesteps`
    timestep_units : {'s', 'min', 'h', 'd', 'a'}, optional
        Units for values specified in the `timesteps` argument. 's' means
        seconds, 'min' means minutes, 'h' means hours, and 'a' means Julian
        years.

    Returns
    -------
    dict
        Dictionary mapping nuclide to an array of time correction factors for
        each time.

    """

    # Determine normalized timesteps and source rates
    timesteps, source_rates = _normalize_timesteps(
        timesteps, source_rates, timestep_units)

    # Calculate decay rate for each nuclide
    decay_rate = np.array([log(2.0) / half_life(x) for x in nuclides])

    n_timesteps = len(timesteps) + 1
    n_nuclides = len(nuclides)

    # Create a 2D array for the time correction factors
    h = np.zeros((n_timesteps, n_nuclides))

    # Precompute all exponential terms with same shape as h
    decay_dt = decay_rate[np.newaxis, :] * timesteps[:, np.newaxis]
    g = np.exp(-decay_dt)
    one_minus_g = -np.expm1(-decay_dt)

    # Apply recurrence relation step by step
    for i in range(len(timesteps)):
        # Eq. (4) in doi:10.1016/j.fusengdes.2019.111399
        h[i + 1] = source_rates[i] * one_minus_g[i] + h[i] * g[i]

    return {nuclides[i]: h[:, i] for i in range(n_nuclides)}


def apply_time_correction(
        tally: openmc.Tally,
        time_correction_factors: dict[str, np.ndarray],
        index: int = -1,
        sum_nuclides: bool = True
) -> openmc.Tally:
    """Apply time correction factors to a tally.

    This function applies the time correction factors at the given index to a
    tally that contains a :class:`~openmc.ParentNuclideFilter`. When
    `sum_nuclides` is True, values over all parent nuclides will be summed,
    leaving a single value for each filter combination.

    Parameters
    ----------
    tally : openmc.Tally
        Tally to apply the time correction factors to
    time_correction_factors : dict
        Time correction factors as returned by :func:`time_correction_factors`
    index : int, optional
        Index of the time of interest. If N timesteps are provided in
        :func:`time_correction_factors`, there are N + 1 times to select from.
        The default is -1 which corresponds to the final time.
    sum_nuclides : bool
        Whether to sum over the parent nuclides

    Returns
    -------
    openmc.Tally
        Derived tally with time correction factors applied

    """
    # Make sure the tally contains a ParentNuclideFilter
    for i_filter, filter in enumerate(tally.filters):
        if isinstance(filter, openmc.ParentNuclideFilter):
            break
    else:
        raise ValueError('Tally must contain a ParentNuclideFilter')

    # Get list of radionuclides based on tally filter
    radionuclides = [str(x) for x in tally.filters[i_filter].bins]
    tcf = np.array([time_correction_factors[x][index] for x in radionuclides])

    # Force tally results to be read and std_dev to be computed
    tally.std_dev

    # Create shallow copy of tally
    new_tally = copy(tally)
    new_tally._filters = copy(tally._filters)

    # Determine number of bins in other filters
    n_bins_before = prod([f.num_bins for f in tally.filters[:i_filter]])
    n_bins_after = prod([f.num_bins for f in tally.filters[i_filter + 1:]])

    # Reshape sum and sum_sq, apply TCF, and sum along that axis
    _, n_nuclides, n_scores = new_tally.shape
    n_radionuclides = len(radionuclides)
    shape = (n_bins_before, n_radionuclides, n_bins_after, n_nuclides, n_scores)
    tally_sum = new_tally.sum.reshape(shape)
    tally_sum_sq = new_tally.sum_sq.reshape(shape)
    tally_mean = new_tally.mean.reshape(shape)
    tally_std_dev = new_tally.std_dev.reshape(shape)

    # Apply TCF, broadcasting to the correct dimensions
    tcf.shape = (1, -1, 1, 1, 1)
    new_tally._sum = tally_sum * tcf
    new_tally._sum_sq = tally_sum_sq * (tcf*tcf)
    new_tally._mean = tally_mean * tcf
    new_tally._std_dev = tally_std_dev * tcf

    shape = (-1, n_nuclides, n_scores)

    if sum_nuclides:
        # Sum over parent nuclides (note that when combining different bins for
        # parent nuclide, we can't work directly on sum_sq)
        new_tally._mean = new_tally.mean.sum(axis=1).reshape(shape)
        new_tally._std_dev = np.linalg.norm(new_tally.std_dev, axis=1).reshape(shape)
        new_tally._derived = True

        # Remove ParentNuclideFilter
        new_tally.filters.pop(i_filter)
    else:
        # Change shape back to (filter combinations, nuclides, scores)
        new_tally._sum.shape = shape
        new_tally._sum_sq.shape = shape
        new_tally._mean.shape = shape
        new_tally._std_dev.shape = shape

    return new_tally


def apply_time_correction_series(
        tally: openmc.Tally,
        time_correction_factors: dict[str, np.ndarray],
        indices: Sequence[int] | None = None,
        sum_nuclides: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply time correction factors to a tally at multiple time indices.

    Vectorized variant of :func:`apply_time_correction` that evaluates a series
    of time indices in a single matrix multiplication. Calling
    :func:`apply_time_correction` in a loop over ``N`` indices deep-copies the
    tally ``N`` times and does ``N`` applications of the
    sum/sum_sq/mean/std_dev arithmetic; this function reads the underlying
    arrays once and folds the radionuclide-axis sum into one matmul, so the
    work is independent of ``N`` on the tally-extraction side.

    Unlike :func:`apply_time_correction`, this returns raw NumPy arrays rather
    than a list of derived :class:`openmc.Tally` objects: constructing ``N``
    derived tallies (each with its own copy of ``_sum``, ``_sum_sq``,
    ``_mean``, and ``_std_dev``) negates the memory advantage on fine-mesh
    tallies. Users who need a ``Tally`` per index can build one from the
    returned arrays.

    Parameters
    ----------
    tally : openmc.Tally
        Tally to apply the time correction factors to. Must contain a
        :class:`~openmc.ParentNuclideFilter`.
    time_correction_factors : dict
        Time correction factors as returned by :func:`time_correction_factors`.
    indices : iterable of int, optional
        Indices into each time correction factor array to evaluate. If None
        (default), every available index is evaluated.
    sum_nuclides : bool, optional
        Whether to sum over the parent nuclides (default True). Matches the
        semantics of :func:`apply_time_correction`: with ``sum_nuclides=True``
        the standard deviation across radionuclides is the L2 norm.

    Returns
    -------
    mean : numpy.ndarray
        Mean values. Shape is ``(n_indices, n_other_filter_bins, n_nuclides,
        n_scores)`` when ``sum_nuclides`` is True (the
        :class:`~openmc.ParentNuclideFilter` axis is collapsed), and
        ``(n_indices, n_filter_bins, n_nuclides, n_scores)`` otherwise (with
        the parent-nuclide bins flattened into the filter-bin axis, same
        layout as :func:`apply_time_correction` produces).
    std_dev : numpy.ndarray
        Standard deviations with the same shape as ``mean``.

    """
    # Locate the ParentNuclideFilter
    for i_filter, f in enumerate(tally.filters):
        if isinstance(f, openmc.ParentNuclideFilter):
            break
    else:
        raise ValueError('Tally must contain a ParentNuclideFilter')

    radionuclides = [str(x) for x in tally.filters[i_filter].bins]
    n_radionuclides = len(radionuclides)

    # Default to every available index
    if indices is None:
        indices = range(len(time_correction_factors[radionuclides[0]]))
    indices = np.asarray(list(indices), dtype=int)
    n_indices = indices.size

    # Build (n_indices, n_radionuclides) TCF matrix
    tcf = np.column_stack(
        [time_correction_factors[nuc][indices] for nuc in radionuclides]
    )

    # Force std_dev to be computed and the underlying arrays to be read
    tally.std_dev

    # Reshape to expose the parent-nuclide axis
    n_bins_before = prod([f.num_bins for f in tally.filters[:i_filter]])
    n_bins_after = prod([f.num_bins for f in tally.filters[i_filter + 1:]])
    _, n_nuclides, n_scores = tally.shape
    shape5 = (n_bins_before, n_radionuclides, n_bins_after, n_nuclides, n_scores)

    mean_5d = tally.mean.reshape(shape5)
    std_dev_5d = tally.std_dev.reshape(shape5)

    if sum_nuclides:
        # Move parent-nuclide axis to position 0 and flatten the rest so a
        # single matmul does the per-index radionuclide sum.
        mean_rf = np.moveaxis(mean_5d, 1, 0).reshape(n_radionuclides, -1)
        var_rf = np.moveaxis(std_dev_5d ** 2, 1, 0).reshape(n_radionuclides, -1)

        mean_out = tcf @ mean_rf
        # Variances combine linearly when factors are squared; sqrt at the end
        # gives the L2 norm matching apply_time_correction.
        std_out = np.sqrt((tcf ** 2) @ var_rf)

        out_shape = (n_indices, n_bins_before * n_bins_after,
                     n_nuclides, n_scores)
        return mean_out.reshape(out_shape), std_out.reshape(out_shape)

    # Per-radionuclide: result keeps the parent-nuclide axis.
    tcf_b = tcf.reshape(n_indices, 1, n_radionuclides, 1, 1, 1)
    mean_out = tcf_b * mean_5d[np.newaxis, ...]
    std_out = tcf_b * std_dev_5d[np.newaxis, ...]

    out_shape = (n_indices, -1, n_nuclides, n_scores)
    return mean_out.reshape(out_shape), std_out.reshape(out_shape)


def prepare_tallies(
        model: openmc.Model,
        nuclides: list[str] | None = None,
        chain_file: str | None = None
) -> list[str]:
    """Prepare tallies for the D1S method.

    This function adds a :class:`~openmc.ParentNuclideFilter` to any tally that
    has a particle filter with a single 'photon' bin.

    Parameters
    ----------
    model : openmc.Model
        Model to prepare tallies for
    nuclides : list of str, optional
        Nuclides to use for the parent nuclide filter. If None, radionuclides
        are determined from :func:`get_radionuclides`.
    chain_file : str, optional
        Chain file to use for inspecting decay data. If None, defaults to
        ``openmc.config['chain_file']``

    Returns
    -------
    list of str
        List of parent nuclides being filtered on

    """
    if nuclides is None:
        nuclides = get_radionuclides(model, chain_file)
    filter = openmc.ParentNuclideFilter(nuclides)

    # Apply parent nuclide filter to any tally that has a particle filter with a
    # single 'photon' bin
    for tally in model.tallies:
        for f in tally.filters:
            if isinstance(f, openmc.ParticleFilter):
                if list(f.bins) == ['photon']:
                    if not tally.contains_filter(openmc.ParentNuclideFilter):
                        tally.filters.append(filter)
                    break
    return nuclides
