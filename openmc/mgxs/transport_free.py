"""Deterministic, transport-free multigroup cross-section collapse.

Collapses continuous-energy nuclear data against an *assumed* weighting flux
with narrow-resonance (Bondarenko) self-shielding -- no Monte Carlo, no
transport solve, no ``nparticles``. Each material is collapsed *directly*
(its own macroscopic flux); per-nuclide multigroup data are never combined,
because multigroup cross sections are flux-weighted averages and do not add
cleanly (see shimwell/openmc#112, design principle 7).

This is the Phase-1 vector-cross-section core (total / absorption / capture /
fission / nu-fission). Scatter matrices and the 0-D slowing-down weighting
(option 3) come later.
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


def collapse_material(material, groups, temperature=294.0, cross_sections=None,
                      self_shield=True):
    """Transport-free macroscopic multigroup cross sections for one material.

    Parameters
    ----------
    material : openmc.Material
    groups : EnergyGroups | str | sequence of float
    temperature : float
        Target temperature [K] (nearest available data temperature is used).
    self_shield : bool
        If True, use the narrow-resonance self-shielded flux phi = w(E)/Sigma_t(E);
        if False, use the unshielded smooth flux phi = w(E) (infinite dilution).

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

    # Weighting flux: smooth part w(E) = 1/E (asymptotic slowing-down),
    # narrow-resonance self-shielded by the material's own total.
    w = 1.0 / np.clip(grid, 1e-11, None)
    phi = w / np.clip(sigma_t, 1e-30, None) if self_shield else w

    reactions = {
        'total': sigma_t,
        'absorption': _macroscopic(incs, dens, temp_str, grid, 101),
        'capture': _macroscopic(incs, dens, temp_str, grid, 102),
        'fission': _macroscopic(incs, dens, temp_str, grid, 18),
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
