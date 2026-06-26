"""Deterministic, transport-free multigroup cross-section collapse.

Collapses continuous-energy nuclear data against an *assumed* weighting flux
with narrow-resonance (Bondarenko) self-shielding -- no Monte Carlo, no
transport solve, no ``nparticles``. Each material is collapsed *directly*
(its own macroscopic flux); per-nuclide multigroup data are never combined,
because multigroup cross sections are flux-weighted averages and do not add
cleanly (see shimwell/openmc#112, design principle 7).

Vector cross sections only (total / absorption / capture / fission). Weighting
is a 1/E (+ optional source) narrow-resonance flux; self-shielding (resolved
resonances plus the unresolved range via probability tables) is always applied.
"""
from __future__ import annotations

import numpy as np

import openmc
import openmc.data
from .groups import EnergyGroups

# numpy >= 2.0 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _as_energy_groups(groups) -> EnergyGroups:
    from openmc.mgxs import GROUP_STRUCTURES  # defined in openmc/mgxs/__init__.py
    if isinstance(groups, EnergyGroups):
        return groups
    if isinstance(groups, str):
        return EnergyGroups(GROUP_STRUCTURES[groups])
    return EnergyGroups(np.asarray(groups, dtype=float))


def _nearest_temperature(inc: "openmc.data.IncidentNeutron", temperature: float) -> str:
    temps = np.array([float(t[:-1]) for t in inc.temperatures])
    return inc.temperatures[int(np.argmin(np.abs(temps - temperature)))]


def _source_pdf(dist, grid):
    """Evaluate a source energy distribution as a normalized PDF on ``grid``."""
    import openmc.stats as st
    pdf = np.zeros_like(grid)
    if isinstance(dist, st.Normal):                       # e.g. openmc.stats.muir()
        mu, sig = float(dist.mean_value), float(dist.std_dev)
        pdf = np.exp(-0.5 * ((grid - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    elif isinstance(dist, st.Discrete):                   # mono lines (e.g. DD 2.45 MeV)
        for xi, pi in zip(np.atleast_1d(dist.x), np.atleast_1d(dist.p)):
            j = int(np.clip(np.searchsorted(grid, xi), 1, len(grid) - 1))
            pdf[j - 1] += pi / max(grid[j] - grid[j - 1], 1e-30)
    elif isinstance(dist, st.Tabular):                    # TT continuum / arbitrary
        pdf = np.interp(grid, dist.x, dist.p, left=0.0, right=0.0)
    elif isinstance(dist, st.Mixture):                    # mixtures
        for p, d in zip(dist.probability, dist.distribution):
            pdf = pdf + p * _source_pdf(d, grid)
    integral = _trapz(pdf, grid)
    return pdf / integral if integral > 0 else pdf


def _macroscopic(incs, dens, temp_str, grid, mt):
    """Macroscopic pointwise xs (1/cm) for reaction ``mt`` on ``grid``; None if absent."""
    total = np.zeros_like(grid)
    present = False
    for nuc, n in dens.items():
        inc = incs[nuc]
        try:
            xs = inc[mt].xs[temp_str[nuc]]
        except KeyError:
            continue
        total += n * xs(grid)
        present = True
    return total if present else None


def _apply_urr(incs, dens, temp_str, grid, sigma_t_smooth, temperature):
    """Unresolved-resonance self-shielding via probability tables (Bondarenko).

    In the unresolved range the pointwise data is the infinitely-dilute (smooth)
    average, so phi = 1/Sigma_t applies *no* self-shielding there. The probability
    tables restore the band structure: for each band b (probability p_b, micro
    total sigma_t,b), the flux ~ 1/(sigma_t,b + sigma_0), giving the self-shielded
    effective micro xs  <sigma_x> = sum_b p_b sigma_x,b/(sigma_t,b+sigma_0)
                                   / sum_b p_b/(sigma_t,b+sigma_0),
    with sigma_0 the per-resonant-nuclide background from the rest of the material.
    Tables here are LSSF=1 factors on the smooth xs (``multiply_smooth``).

    Returns {mt: macroscopic delta array (1/cm)} to ADD to the dilute macroscopic
    cross sections, for mt in (1, 101, 102).
    """
    delta = {1: np.zeros_like(grid), 101: np.zeros_like(grid), 102: np.zeros_like(grid)}
    for nuc, n in dens.items():
        inc = incs[nuc]
        urr = getattr(inc, 'urr', None)
        if not urr:
            continue
        cand = [t for t in urr if urr.get(t) is not None and t in inc.temperatures]
        if not cand:
            continue
        ts = min(cand, key=lambda t: abs(float(t[:-1]) - temperature))
        pt = urr[ts]
        if not getattr(pt, 'multiply_smooth', False):
            continue                                   # only LSSF=1 factor tables
        e = np.asarray(pt.energy, dtype=float)
        tab = np.asarray(pt.table, dtype=float)        # [nE, 6, nbands]
        idx = np.where((grid >= e[0]) & (grid <= e[-1]))[0]
        if idx.size == 0:
            continue
        st = inc[1].xs[ts](grid)                        # smooth micro total
        sc = inc[102].xs[ts](grid)                      # smooth micro capture
        try:
            sa = inc[101].xs[ts](grid)
        except Exception:
            sa = sc
        for gi in idx:
            j = int(np.clip(np.searchsorted(e, grid[gi]), 1, len(e) - 1))
            row = tab[j - 1] if abs(e[j - 1] - grid[gi]) <= abs(e[j] - grid[gi]) else tab[j]
            p = np.diff(np.concatenate(([0.0], row[0])))    # band probabilities
            sig0 = (sigma_t_smooth[gi] - n * st[gi]) / n    # micro background (others)
            if sig0 < 0:
                sig0 = 0.0
            stb = st[gi] * row[1]                            # band micro total
            w = p / (stb + sig0)
            denom = w.sum()
            if denom <= 0:
                continue
            st_eff = float((w * stb).sum() / denom)
            sc_eff = float((w * (sc[gi] * row[4])).sum() / denom)
            fc = sc_eff / sc[gi] if sc[gi] > 0 else 1.0     # shield absorption like capture
            delta[1][gi]   += n * (st_eff - st[gi])
            delta[102][gi] += n * (sc_eff - sc[gi])
            delta[101][gi] += n * sa[gi] * (fc - 1.0)
    return delta


def collapse_material(material, groups, temperature=294.0, cross_sections=None,
                      source=None):
    """Transport-free macroscopic multigroup cross sections for one material.

    Parameters
    ----------
    material : openmc.Material
    groups : EnergyGroups | str | sequence of float
    temperature : float
        Target temperature [K] (nearest available data temperature is used).

    Self-shielding (resolved resonances via phi = w/Sigma_t, plus the unresolved
    range via probability tables) is always applied: a real material is never at
    infinite dilution, so there is no knob to turn it off.

    Returns
    -------
    dict with 'group_edges' (eV, ascending) and macroscopic group XS arrays
    (1/cm), ordered group 1 = highest energy (OpenMC convention).
    """
    groups = _as_energy_groups(groups)
    edges = np.asarray(groups.group_edges, dtype=float)
    emin, emax = edges[0], edges[-1]

    if cross_sections is None:
        cross_sections = openmc.config['cross_sections']
    datalib = openmc.data.DataLibrary.from_xml(cross_sections)

    dens = material.get_nuclide_atom_densities()  # {nuclide: atom/b-cm}

    incs, temp_str, grids = {}, {}, [edges]
    for nuc in dens:
        entry = datalib.get_by_material(nuc, data_type='neutron')
        inc = openmc.data.IncidentNeutron.from_hdf5(entry['path'])
        incs[nuc] = inc
        ts = _nearest_temperature(inc, temperature)
        temp_str[nuc] = ts
        grids.append(np.asarray(inc.energy[ts]))

    grid = np.unique(np.concatenate(grids))
    grid = grid[(grid >= emin) & (grid <= emax)]

    sigma_t = _macroscopic(incs, dens, temp_str, grid, 1)
    if sigma_t is None:
        raise ValueError("no total cross section (MT=1) found for material")

    sigma_a = _macroscopic(incs, dens, temp_str, grid, 101)
    sigma_c = _macroscopic(incs, dens, temp_str, grid, 102)
    sigma_f = _macroscopic(incs, dens, temp_str, grid, 18)

    # Unresolved-resonance self-shielding (probability tables): correct the dilute
    # total / absorption / capture band-by-band in the URR before weighting. Always
    # applied -- it is part of self-shielding, which a real material always has.
    d = _apply_urr(incs, dens, temp_str, grid, sigma_t, temperature)
    sigma_t = sigma_t + d[1]
    if sigma_a is not None:
        sigma_a = sigma_a + d[101]
    if sigma_c is not None:
        sigma_c = sigma_c + d[102]

    # Narrow-resonance weighting flux: phi = w(E)/Sigma_t(E). The smooth part
    # w = 1/E (asymptotic slowing-down) is optionally sharpened in the fast groups
    # by the source spectrum, then self-shielded by the material's own
    # (URR-corrected) total.
    w = 1.0 / np.clip(grid, 1e-11, None)
    if source is not None:
        w = w + _source_pdf(source, grid)
    phi = w / np.clip(sigma_t, 1e-30, None)

    reactions = {
        'total': sigma_t,
        'absorption': sigma_a,
        'capture': sigma_c,
        'fission': sigma_f,
    }

    G = groups.num_groups
    out = {'group_edges': edges}
    for name, sig in reactions.items():
        if sig is None:
            continue
        sig_g = np.zeros(G)
        for g in range(G):           # g: ascending in energy
            lo, hi = edges[g], edges[g + 1]
            m = (grid >= lo) & (grid <= hi)
            if m.sum() < 2:
                continue
            x, p, s = grid[m], phi[m], sig[m]
            den = _trapz(p, x)
            sig_g[g] = _trapz(s * p, x) / den if den > 0 else 0.0
        # OpenMC orders groups with group 1 = highest energy → reverse
        out[name] = sig_g[::-1]
    return out
