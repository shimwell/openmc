"""Recovered slowing-down weighting flux (option 3) for the scatter-matrix weighting experiment."""
import numpy as np
import openmc, openmc.data
from openmc.mgxs.transport_free import _source_pdf
_trapz = getattr(np, "trapezoid", None) or np.trapz

def _outgoing(dist):
    """Classify a secondary-neutron energy distribution for the transfer kernel.

    Returns ('delta', threshold, mass_ratio) for a discrete inelastic level
    (E_out = mass_ratio*(E_in - threshold)), or ('tab', incident_energies,
    [Tabular,...]) for a tabulated continuum / (n,xn) distribution, or None.
    """
    try:
        from openmc.data import (UncorrelatedAngleEnergy, CorrelatedAngleEnergy,
                                 LevelInelastic)
    except Exception:
        return None
    if isinstance(dist, CorrelatedAngleEnergy):
        return ('tab', np.asarray(dist.energy, dtype=float), dist.energy_out)
    ed = dist.energy if isinstance(dist, UncorrelatedAngleEnergy) else getattr(dist, 'energy', None)
    if isinstance(ed, LevelInelastic):
        return ('delta', float(ed.threshold), float(ed.mass_ratio))
    if ed is not None and hasattr(ed, 'energy_out') and hasattr(ed, 'energy'):
        return ('tab', np.asarray(ed.energy, dtype=float), ed.energy_out)
    return None


def _slowing_down_weight(incs, dens, temp_str, fine_grid, sigma_t_fine,
                         source, per_decade=40):
    """0-D infinite-medium slowing-down weighting flux on ``fine_grid``.

    Solves the energy-domain neutron balance
        Sigma_t(E) phi(E) = S(E) + sum_r integral Sigma_s,r(E') f_r(E'->E) phi(E') dE'
    on a coarse lethargy grid (the slowing-down *source* is smooth), using real
    energy-transfer kernels from ``openmc.data``: analytic elastic (MT=2),
    discrete inelastic levels (MT=51-90), and tabulated continuum / (n,xn)
    (MT=91/16/17). Strictly down-scatter -> a single high->low energy sweep is
    exact. The smooth slowing-down source is then divided by the *fine* total to
    reintroduce resonance self-shielding. No transport, no Monte Carlo.

    Thermal up-scatter (S(alpha,beta)) is not modelled -> valid in the
    fast/epithermal range (the populated range for fast fusion shields).
    """
    emin, emax = fine_grid[0], fine_grid[-1]
    nb = int(max(per_decade * np.log10(emax / emin), 40))
    be = np.logspace(np.log10(emin), np.log10(emax), nb + 1)   # bin edges
    cg = np.sqrt(be[:-1] * be[1:])                              # bin centres
    nb = len(cg)

    # dilute (1/E-weighted) bin-averaged total, smooth -> no erratic resonance sampling
    w = 1.0 / np.clip(fine_grid, 1e-11, None)
    def _cum(y):
        c = np.zeros_like(fine_grid)
        c[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(fine_grid))
        return c
    Ni = np.interp(be, fine_grid, _cum(sigma_t_fine * w))
    Di = np.interp(be, fine_grid, _cum(w))
    sigt_c = np.diff(Ni) / np.clip(np.diff(Di), 1e-300, None)

    # transfer matrix M[l, k] = scatter rate into bin l per unit flux in bin k
    M = np.zeros((nb, nb))
    for nuc, dn in dens.items():
        inc = incs[nuc]
        ts = temp_str[nuc]
        alpha = ((inc.atomic_weight_ratio - 1.0) / (inc.atomic_weight_ratio + 1.0)) ** 2
        for mt, r in inc.reactions.items():
            is_el = (mt == 2)
            if not (is_el or (51 <= mt <= 91) or mt in (16, 17)):
                continue
            try:
                sig = dn * r.xs[ts](cg)
            except Exception:
                continue
            if not np.any(sig > 0):
                continue
            if is_el:
                lo = alpha * cg
                for k in range(nb):
                    if sig[k] <= 0:
                        continue
                    width = cg[k] - lo[k]
                    if width <= 0:                       # alpha ~ 1 (heavy): no loss
                        M[k, k] += sig[k]
                        continue
                    ov = np.clip(np.minimum(be[1:], cg[k]) - np.maximum(be[:-1], lo[k]), 0, None)
                    M[:, k] += sig[k] * ov / width
                continue
            prods = [p for p in r.products if p.particle == 'neutron']
            if not prods:
                continue
            try:
                og = _outgoing(prods[0].distribution[0])
            except Exception:
                og = None
            if og is None:
                continue
            try:
                mult = np.atleast_1d(np.asarray(prods[0].yield_(cg), dtype=float))
                if mult.size == 1:
                    mult = np.full(nb, float(mult[0]))
            except Exception:
                mult = np.ones(nb)
            if og[0] == 'delta':
                thr, mr = og[1], og[2]
                eout = mr * (cg - thr)
                for k in range(nb):
                    if sig[k] <= 0 or cg[k] <= thr or eout[k] < be[0]:
                        continue
                    l = min(max(np.searchsorted(be, eout[k]) - 1, 0), nb - 1)
                    M[l, k] += sig[k] * mult[k]
            else:                                        # 'tab'
                ein, eos = og[1], og[2]
                for k in range(nb):
                    if sig[k] <= 0 or cg[k] < ein[0]:
                        continue
                    t = eos[min(np.searchsorted(ein, cg[k]), len(eos) - 1)]
                    cx, cp = np.asarray(t.x), np.asarray(t.p)
                    cc = np.zeros_like(cx)
                    cc[1:] = np.cumsum(0.5 * (cp[1:] + cp[:-1]) * np.diff(cx))
                    if cc[-1] <= 0:
                        continue
                    Wb = np.diff(np.interp(be, cx, cc / cc[-1], left=0.0, right=1.0))
                    M[:, k] += sig[k] * mult[k] * Wb

    # external (or generic top-energy) source on the coarse grid
    S = _source_pdf(source, cg) if source is not None else np.zeros(nb)
    if S.sum() <= 0:
        S = np.zeros(nb)
        S[-1] = 1.0

    # exact downward sweep (strictly down-scatter)
    phi = np.zeros(nb)
    for k in range(nb - 1, -1, -1):
        inscat = M[k, k + 1:].dot(phi[k + 1:]) if k + 1 < nb else 0.0
        denom = sigt_c[k] - M[k, k]                      # remove within-bin self-scatter
        phi[k] = (S[k] + inscat) / (denom if denom > 1e-30 else max(sigt_c[k], 1e-30))

    qtot = phi * sigt_c                                  # smooth slowing-down source
    good = qtot > 0
    if good.sum() < 2:                                   # degenerate -> fall back to 1/E
        return (1.0 / np.clip(fine_grid, 1e-11, None)) / np.clip(sigma_t_fine, 1e-30, None)
    qf = np.exp(np.interp(np.log(fine_grid), np.log(cg[good]), np.log(qtot[good])))
    return qf / np.clip(sigma_t_fine, 1e-30, None)       # self-shield on the fine total



def _slowing_down_flux_fine(incs, dens, temp_str, grid, sigma_t, source):
    """PROPER 0-D infinite-medium slowing-down flux, solved on the FINE resonance grid.

    Solves
        Sigma_t(E) phi(E) = S(E)
                          + sum_r int_E^{E/alpha_r} Sigma_s,r(E')/((1-alpha_r)E') phi(E') dE'
    by an EXACT high->low energy sweep on the pointwise grid, so the scattering
    source carries phi's OWN resonance dips self-consistently. THIS is what
    distinguishes it from narrow-resonance (phi = source/Sigma_t) -- it captures the
    shallower flux depression at wide scattering resonances (the wide/intermediate
    resonance effect) with no Goldstein-Cohen lambda. Isotropic-CM elastic kernel
    (the standard slowing-down kernel); elastic-only (resonance region is elastic-
    dominated, below the inelastic thresholds). Down-scatter only.
    """
    N = len(grid)
    dE = np.empty(N)
    dE[1:-1] = 0.5 * (grid[2:] - grid[:-2]); dE[0] = 0.5*(grid[1]-grid[0]); dE[-1] = 0.5*(grid[-1]-grid[-2])
    S = _source_pdf(source, grid).astype(float) if source is not None else 1.0/np.clip(grid, 1e-11, None)

    # per-nuclide: coefdE_j = Sigma_s,el(E_j)/((1-alpha)E_j) * dE_j  (the quadrature weight),
    # and jhi[i] = first grid index strictly above E_i/alpha (the slowing-down upper limit).
    kerns = []
    for nuc, n in dens.items():
        inc = incs[nuc]; ts = temp_str[nuc]; A = inc.atomic_weight_ratio
        alpha = ((A - 1.0) / (A + 1.0)) ** 2
        if alpha >= 1.0 - 1e-9:
            continue
        try:
            ss = n * inc[2].xs[ts](grid)
        except Exception:
            continue
        if not np.any(ss > 0):
            continue
        coefdE = ss / ((1.0 - alpha) * np.clip(grid, 1e-30, None)) * dE
        jhi = np.searchsorted(grid, grid / alpha, side='right')      # vectorised upper limits
        kerns.append((coefdE, jhi))

    phi = np.zeros(N)
    for i in range(N - 1, -1, -1):
        rhs = S[i]; self_c = 0.0
        for coefdE, jhi in kerns:
            jh = jhi[i]
            if jh > i + 1:
                rhs += coefdE[i + 1:jh] @ phi[i + 1:jh]              # in-scatter from above (known)
            self_c += coefdE[i]                                      # E'=E_i self term -> LHS
        denom = sigma_t[i] - self_c
        phi[i] = rhs / denom if denom > 1e-30 else rhs / max(sigma_t[i], 1e-30)
    return phi
