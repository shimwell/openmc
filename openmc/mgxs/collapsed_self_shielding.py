"""Collapsed, self-shielded multigroup cross-section generation.

Generates multigroup cross sections in the style of NJOY (GROUPR) and FISPACT:
collapse (group-average) continuous-energy nuclear data against an *assumed*
weighting flux, with narrow-resonance (Bondarenko) resonance self-shielding --
no Monte Carlo, no transport solve, no ``nparticles``. Each material is
collapsed *directly*
(its own macroscopic flux); per-nuclide multigroup data are never combined,
because multigroup cross sections are flux-weighted averages and do not add
cleanly (see shimwell/openmc#112, design principle 7).

``collapse_material`` gives the vector cross sections (total / absorption /
capture / fission); ``scatter_matrix`` gives the P0 group-to-group scattering
matrix (elastic with the real CM angular distribution, discrete inelastic levels,
and unit-base-interpolated continuum / (n,xn) with multiplicity) -- together a
complete MG library for a P0 solver such as random ray. Weighting is a 1/E
(+ optional source) narrow-resonance flux; self-shielding (resolved resonances
plus the unresolved range via probability tables) is always applied.
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


def _urr_elastic_factor(incs, dens, temp_str, grid, sigma_t_smooth, temperature):
    """Per-nuclide micro elastic self-shielding factor f_el(E) = <sigma_el>/sigma_el,smooth
    in the unresolved range (1.0 elsewhere), from the same probability-table band average
    as :func:`_apply_urr` but for the elastic channel (table column 2). Lets the scatter
    matrix self-shield its elastic consistently with the URR-corrected vector total --
    otherwise the scatter row-sum is the dilute (too-high) elastic across the URR.
    """
    fac = {}
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
            continue
        e = np.asarray(pt.energy, float); tab = np.asarray(pt.table, float)
        idx = np.where((grid >= e[0]) & (grid <= e[-1]))[0]
        if idx.size == 0:
            continue
        st = inc[1].xs[ts](grid); se = inc[2].xs[ts](grid)
        f = np.ones_like(grid)
        for gj in idx:
            if se[gj] <= 0:
                continue
            j = int(np.clip(np.searchsorted(e, grid[gj]), 1, len(e) - 1))
            row = tab[j - 1] if abs(e[j - 1] - grid[gj]) <= abs(e[j] - grid[gj]) else tab[j]
            p = np.diff(np.concatenate(([0.0], row[0])))
            sig0 = (sigma_t_smooth[gj] - n * st[gj]) / n
            if sig0 < 0:
                sig0 = 0.0
            wgt = p / (st[gj] * row[1] + sig0)
            den = wgt.sum()
            if den <= 0:
                continue
            f[gj] = float((wgt * (se[gj] * row[2])).sum() / den) / se[gj]
        fac[nuc] = f
    return fac


def collapse_material(material, groups, temperature=294.0, cross_sections=None,
                      source=None):
    """Collapsed, self-shielded macroscopic multigroup cross sections for one material.

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


# ----------------------------------------------------------------------------
# P0 group-to-group scattering matrix (deterministic)
# ----------------------------------------------------------------------------

def _curve_frac(x, p, edges):
    """Group-integrate a lin-lin density (x, p) over ascending edges -> len-G, normalized."""
    x = np.asarray(x, float); p = np.asarray(p, float)
    if x.size < 2:
        return None
    cc = np.zeros_like(x); cc[1:] = np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))
    tot = cc[-1]
    if tot <= 0:
        return None
    return np.diff(np.interp(edges, x, cc, left=0.0, right=tot)) / tot


def _tab_frac(t, edges):
    """Group fractions for an ``openmc.stats.Tabular`` outgoing distribution."""
    try:
        cc = np.asarray(t.cdf(), float); x = np.asarray(t.x, float)
        if cc[-1] <= 0:
            return None
        return np.diff(np.interp(edges, x, cc, left=0.0, right=cc[-1])) / cc[-1]
    except Exception:
        return _curve_frac(getattr(t, 'x', []), getattr(t, 'p', []), edges)


def _unitbase_frac(d_lo, d_hi, f, edges):
    """Unit-base interpolation of two Tabulars at fraction ``f``, group-integrated.

    Both outgoing distributions are mapped to the unit interval, blended there
    (valid because the two span different outgoing-energy ranges), then mapped back
    so every outgoing energy stays within the interpolated kinematic range.
    """
    xl, pl = np.asarray(d_lo.x, float), np.asarray(d_lo.p, float)
    xh, ph = np.asarray(d_hi.x, float), np.asarray(d_hi.p, float)
    if xl.size < 2 or xh.size < 2:
        return _tab_frac(d_lo if f < 0.5 else d_hi, edges)
    Ll, Lh = xl[-1] - xl[0], xh[-1] - xh[0]
    if Ll <= 0 or Lh <= 0:
        return _tab_frac(d_lo if f < 0.5 else d_hi, edges)
    E0 = xl[0] + f * (xh[0] - xl[0]); E1 = xl[-1] + f * (xh[-1] - xl[-1])
    if E1 <= E0:
        return None
    ul = (xl - xl[0]) / Ll; uh = (xh - xh[0]) / Lh
    u = np.union1d(ul, uh)
    pu = (1 - f) * np.interp(u, ul, pl * Ll, left=0, right=0) + f * np.interp(u, uh, ph * Lh, left=0, right=0)
    return _curve_frac(E0 + u * (E1 - E0), pu, edges)


def _incident_code(dist, k):
    """Interpolation code on the incident interval starting at index ``k`` (2=lin-lin, 1=histogram)."""
    bps = getattr(dist, 'breakpoints', None); itp = getattr(dist, 'interpolation', None)
    if bps is None or itp is None:
        return 2
    for r, bp in enumerate(np.atleast_1d(bps)):
        if k + 1 <= bp:
            return int(np.atleast_1d(itp)[r])
    return 2


def _tabdist(dist):
    """Return the object exposing .energy (incident) + .energy_out (Tabulars), or None."""
    if hasattr(dist, 'energy_out') and hasattr(dist, 'energy') and \
            not isinstance(getattr(dist, 'energy'), openmc.data.LevelInelastic):
        return dist
    ed = getattr(dist, 'energy', None)
    if ed is not None and hasattr(ed, 'energy_out') and hasattr(ed, 'energy'):
        return ed
    return None


def _unitbase_anchors(td, edges, per=4):
    """Precompute unit-base group fractions at sub-sampled incident energies."""
    ein = np.asarray(td.energy, float); eos = td.energy_out; K = len(eos)
    Ea, GF = [], []
    G = len(edges) - 1
    for k in range(K - 1):
        hist = _incident_code(td, k) == 1
        for s in range(per):
            f = s / per
            gf = _tab_frac(eos[k], edges) if hist else _unitbase_frac(eos[k], eos[k + 1], f, edges)
            Ea.append(ein[k] + f * (ein[k + 1] - ein[k])); GF.append(gf if gf is not None else np.zeros(G))
    gf = _tab_frac(eos[-1], edges)
    Ea.append(ein[-1]); GF.append(gf if gf is not None else np.zeros(G))
    return np.array(Ea), np.array(GF)


def _freegas_gf(E, A, kT, edges, nv=28, nmu=14):
    """Free-gas (ideal-gas) elastic energy-transfer group fractions at incident energy E.

    Quadrature over the target Maxwellian (speed v_t, cosine mu_t); for each target the
    isotropic-CM elastic scatter gives the lab outgoing energy uniform over
    [0.5(Vcm-w)^2, 0.5(Vcm+w)^2]. Captures thermal up-scatter and broadening that the
    static (target-at-rest) kernel misses; reduces to Wigner-Wilkins for A=1 (verified
    to ~1-2%). E and kT in eV; speeds in sqrt(eV) with m_n = 1.
    """
    vn = np.sqrt(2.0 * E); vth = np.sqrt(2.0 * kT / max(A, 1e-9))
    vt = np.linspace(0.02 * vth, 6.0 * vth, nv)
    wv = vt**2 * np.exp(-A * vt**2 / (2.0 * kT))
    mu = np.linspace(-1.0, 1.0, nmu)
    VT = vt[:, None]; MU = mu[None, :]
    vrel = np.sqrt(np.clip(vn*vn + VT*VT - 2*vn*VT*MU, 0, None))
    Vcm = np.sqrt(np.clip(vn*vn + A*A*VT*VT + 2*A*vn*VT*MU, 0, None)) / (A + 1.0)
    w = (A / (A + 1.0)) * vrel
    Elo = (0.5 * (Vcm - w)**2).ravel(); Ehi = (0.5 * (Vcm + w)**2).ravel()
    wgt = ((wv[:, None]) * np.ones_like(mu)[None, :] * vrel).ravel()
    rng = np.clip(Ehi - Elo, 1e-30, None); tot = wgt.sum()
    if tot <= 0:
        return None
    cdf = (np.clip((edges[:, None] - Elo[None, :]) / rng[None, :], 0, 1) * wgt[None, :]).sum(1) / tot
    return np.diff(cdf)


def scatter_matrix(material, groups, temperature=294.0, cross_sections=None, source=None,
                   return_p1=False, thermal=True):
    """Deterministic P0 group-to-group scattering matrix for one material.

    Returns the macroscopic Sigma_s,g->g' (1/cm) as an ``[G_in, G_out]`` array in
    OpenMC ordering (group 1 = highest energy), matching ``openmc.XSdata`` P0
    (Legendre order 0). Kernels: elastic with the real CM angular distribution,
    discrete inelastic levels spread over their lab energy range, and unit-base
    interpolated continuum / (n,xn) distributions summed over all neutron products
    with their multiplicity. No Monte Carlo. Same weighting flux as
    :func:`collapse_material`.
    """
    groups = _as_energy_groups(groups)
    edges = np.asarray(groups.group_edges, float)
    emin, emax = edges[0], edges[-1]; G = groups.num_groups
    if cross_sections is None:
        cross_sections = openmc.config['cross_sections']
    datalib = openmc.data.DataLibrary.from_xml(cross_sections)
    dens = material.get_nuclide_atom_densities()
    incs, temp_str, grids = {}, {}, [edges]
    for nuc in dens:
        entry = datalib.get_by_material(nuc, data_type='neutron')
        inc = openmc.data.IncidentNeutron.from_hdf5(entry['path']); incs[nuc] = inc
        ts = _nearest_temperature(inc, temperature); temp_str[nuc] = ts
        grids.append(np.asarray(inc.energy[ts]))
    grid = np.unique(np.concatenate(grids)); grid = grid[(grid >= emin) & (grid <= emax)]
    # panel sub-grid: interior points in every group so the incoming integration
    # (esp. the elastic in-group vs down-scatter split) is resolved where data is sparse.
    sub = np.concatenate([np.geomspace(edges[g], edges[g + 1], 17)[1:-1] for g in range(G)])
    grid = np.unique(np.concatenate([grid, sub]))

    sigma_t = _macroscopic(incs, dens, temp_str, grid, 1)
    sigma_t = sigma_t + _apply_urr(incs, dens, temp_str, grid, sigma_t, temperature)[1]
    w = 1.0 / np.clip(grid, 1e-11, None)
    if source is not None:
        w = w + _source_pdf(source, grid)
    phi = w / np.clip(sigma_t, 1e-30, None)
    fel = _urr_elastic_factor(incs, dens, temp_str, grid, sigma_t, temperature)  # URR elastic self-shielding
    dwid = np.empty_like(grid)
    dwid[1:-1] = 0.5 * (grid[2:] - grid[:-2]); dwid[0] = 0.5*(grid[1]-grid[0]); dwid[-1] = 0.5*(grid[-1]-grid[-2])
    wt = phi * dwid
    gi = np.clip(np.searchsorted(edges, grid, side='right') - 1, 0, G - 1)
    M = np.zeros((G, G)); denom = np.zeros(G); np.add.at(denom, gi, wt)
    M1 = np.zeros((G, G)) if return_p1 else None             # P1 (mu_lab-weighted) elastic outscatter

    def overlap(lo, hi):
        if hi <= lo:
            return None
        return np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None) / (hi - lo)

    for nuc, n in dens.items():
        inc = incs[nuc]; ts = temp_str[nuc]; A = inc.atomic_weight_ratio; a1 = (A + 1.0) ** 2
        for mt, r in inc.reactions.items():
            is_el = (mt == 2)
            if not (is_el or (51 <= mt <= 91) or mt in (16, 17)):
                continue
            try:
                xs = r.xs[ts](grid)
            except Exception:
                continue
            if is_el and nuc in fel:                 # self-shield elastic in the URR
                xs = xs * fel[nuc]
            nz = np.nonzero(xs > 0)[0]
            if nz.size == 0:
                continue
            src = n * xs * wt
            if is_el:                                            # elastic: CM angular distribution
                try:
                    ang = r.products[0].distribution[0].angle; aE = np.asarray(ang.energy)
                except Exception:
                    ang = None
                kT = 8.617e-5 * temperature                       # free-gas thermal for light nuclides
                E_TH = 400.0 * kT                                  # = OpenMC free_gas_threshold (default 400 kT);
                do_fg = thermal and A <= 20.0                      # when model-driven, read settings.free_gas_threshold
                tgf = {}
                if do_fg:
                    mid = np.sqrt(edges[:-1] * edges[1:])
                    for g in np.where(mid < E_TH)[0]:
                        gf = _freegas_gf(mid[g], A, kT, edges)
                        if gf is not None:
                            tgf[int(g)] = gf
                for i in nz:
                    E = grid[i]
                    if do_fg and E < E_TH and gi[i] in tgf:       # thermal: free-gas energy transfer
                        M[gi[i]] += src[i] * tgf[gi[i]]; continue
                    mu = fp = None
                    if ang is not None:
                        t = ang.mu[min(np.searchsorted(aE, E), len(aE) - 1)]
                        if hasattr(t, 'x') and hasattr(t, 'p'):
                            mu = np.asarray(t.x, float); fp = np.asarray(t.p, float)
                    if mu is None or mu.size < 2:
                        mu = np.linspace(-1, 1, 33); fp = np.full(33, 0.5)
                    if mu[0] > mu[-1]:
                        mu = mu[::-1]; fp = fp[::-1]
                    # E_out is monotonic in mu, so spread f(mu) SMOOTHLY across outgoing
                    # groups via the angular CDF -- not one delta per mu point, which piles
                    # hydrogen's wide (mu~-1 -> E_out~0) down-scatter into the lowest group.
                    eout = E * (A*A + 2*A*mu + 1.0) / a1
                    cmu = np.concatenate(([0.0], np.cumsum(0.5*(fp[1:]+fp[:-1])*np.diff(mu))))
                    if cmu[-1] <= 0:
                        M[gi[i], gi[i]] += src[i]; continue
                    M[gi[i]] += src[i] * np.diff(np.interp(edges, eout, cmu/cmu[-1], left=0.0, right=1.0))
                    if M1 is not None:                        # mu_lab = (1+A mu)/sqrt(A^2+2A mu+1)
                        mulab = (1.0 + A*mu) / np.sqrt(A*A + 2*A*mu + 1.0)
                        cm1 = np.concatenate(([0.0], np.cumsum(0.5*(fp[1:]*mulab[1:]+fp[:-1]*mulab[:-1])*np.diff(mu))))
                        M1[gi[i]] += src[i] * np.diff(np.interp(edges, eout, cm1, left=0.0, right=cm1[-1])) / cmu[-1]
                continue
            for prod in r.products:                              # inelastic: all neutron products
                if prod.particle != 'neutron':
                    continue
                try:
                    yld = np.atleast_1d(np.asarray(prod.yield_(grid), float))
                    if yld.size == 1:
                        yld = np.full(len(grid), float(yld[0]))
                except Exception:
                    yld = np.ones(len(grid))
                dists = prod.distribution; nd = len(dists)
                appl = getattr(prod, 'applicability', None)
                for k, dist in enumerate(dists):
                    if nd > 1 and appl and k < len(appl):
                        try:
                            app = np.atleast_1d(np.asarray(appl[k](grid), float))
                            if app.size == 1:
                                app = np.full(len(grid), float(app[0]))
                        except Exception:
                            app = np.ones(len(grid))
                    else:
                        app = np.ones(len(grid))
                    ed = getattr(dist, 'energy', None)
                    if isinstance(ed, openmc.data.LevelInelastic):      # discrete level
                        thr, mr = float(ed.threshold), float(ed.mass_ratio)
                        for i in nz:
                            E = grid[i]
                            if E <= thr:
                                continue
                            ecm = mr * (E - thr)
                            if ecm <= 0:
                                continue
                            base = ecm + E / a1; amp = 2.0 * np.sqrt(E * ecm) / (A + 1.0)
                            of = overlap(base - amp, base + amp); wgt = src[i] * yld[i] * app[i]
                            if of is None:
                                M[gi[i], min(max(np.searchsorted(edges, base) - 1, 0), G - 1)] += wgt
                            else:
                                M[gi[i]] += wgt * of
                        continue
                    td = _tabdist(dist)                                  # continuum / (n,xn)
                    if td is None:
                        continue
                    Ea, GF = _unitbase_anchors(td, edges)
                    ein0 = float(np.asarray(td.energy, float)[0])
                    for i in nz:
                        E = grid[i]
                        if E < ein0:
                            continue
                        j = np.searchsorted(Ea, E)
                        if j <= 0:
                            gf = GF[0]
                        elif j >= len(Ea):
                            gf = GF[-1]
                        else:
                            fr = (E - Ea[j-1]) / (Ea[j] - Ea[j-1]); gf = (1 - fr) * GF[j-1] + fr * GF[j]
                        M[gi[i]] += src[i] * yld[i] * app[i] * gf

    # fold upscatter artifacts into the diagonal, but keep real thermal up-scatter (free-gas, below E_TH)
    _midf = np.sqrt(edges[:-1] * edges[1:]); _eth = 400.0 * 8.617e-5 * temperature
    for g in range(G - 1):
        if thermal and _midf[g] < _eth:
            continue
        up = M[g, g+1:].sum()
        if up:
            M[g, g] += up; M[g, g+1:] = 0.0
    M = M / np.clip(denom[:, None], 1e-30, None)
    if return_p1:
        # P1 outscatter moment Sigma_s1,g per group, for a transport-corrected (TC-P0)
        # library: emit sigma_tr = sigma_t - Sigma_s1 and subtract Sigma_s1 from the
        # in-group diagonal. Random ray expects the correction applied here, not in the solver.
        sigma_s1 = (M1.sum(1) / np.clip(denom, 1e-30, None))[::-1]
        return M[::-1, ::-1], sigma_s1
    return M[::-1, ::-1]                                      # -> OpenMC ordering (group 1 = high E)
