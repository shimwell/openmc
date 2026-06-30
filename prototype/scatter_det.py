"""Deterministic P0 scatter matrix Sigma_s,g'->g (transport-free), refined per the
inelastic-kernel research spec. Output matches openmc.XSdata: [G_in, G_out], group 1
= high E, P0, includes (n,xn) yield. Same NR+URR weighting flux as the vector-XS collapse.

Kernels:
  elastic (MT2)        : real CM angular distribution, E_out=E*(A^2+2A mu+1)/(A+1)^2
  discrete level (51-90): spread over lab range [base-amp, base+amp] (CM angular fold),
                          base=E_cm+E/(A+1)^2, amp=2*sqrt(E*E_cm)/(A+1), E_cm=mr*(E-thr)
  continuum/(n,xn)     : UNIT-BASE interpolation of secondary-energy dist between incident
                          energies; sum over ALL product distributions * applicability * yield
  upscatter artifacts folded into the in-group (diagonal) element.
"""
import time, numpy as np, openmc, openmc.data
from openmc.mgxs.transport_free import (_as_energy_groups, _nearest_temperature,
                                        _source_pdf, _macroscopic, _apply_urr)


def _curve_frac(x, p, edges):
    """Group-integrate a lin-lin density (x,p) over ascending group edges -> len-G, normalized."""
    x = np.asarray(x, float); p = np.asarray(p, float)
    if x.size < 2:
        return None
    cc = np.zeros_like(x); cc[1:] = np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))
    tot = cc[-1]
    if tot <= 0:
        return None
    ce = np.interp(edges, x, cc, left=0.0, right=tot)
    return np.diff(ce) / tot


def _tab_frac(t, edges):
    """Group fractions for an openmc.stats.Tabular outgoing distribution."""
    try:
        cc = np.asarray(t.cdf(), float); x = np.asarray(t.x, float)
        if cc[-1] <= 0:
            return None
        ce = np.interp(edges, x, cc, left=0.0, right=cc[-1])
        return np.diff(ce) / cc[-1]
    except Exception:
        return _curve_frac(getattr(t, 'x', []), getattr(t, 'p', []), edges)


def _unitbase_frac(d_lo, d_hi, f, edges):
    """Unit-base interpolation of two Tabulars at fraction f, group-integrated."""
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
    ul = (xl - xl[0]) / Ll; uh = (xh - xh[0]) / Lh          # unit coords
    u = np.union1d(ul, uh)
    pu = (1 - f) * np.interp(u, ul, pl * Ll, left=0, right=0) + f * np.interp(u, uh, ph * Lh, left=0, right=0)
    E = E0 + u * (E1 - E0)                                  # back to physical outgoing energy
    return _curve_frac(E, pu, edges)


def _incident_code(dist, k):
    """Interpolation code on incident interval starting at index k (2=lin-lin default; 1=histogram)."""
    bps = getattr(dist, 'breakpoints', None); itp = getattr(dist, 'interpolation', None)
    if bps is None or itp is None:
        return 2
    for r, bp in enumerate(np.atleast_1d(bps)):
        if k + 1 <= bp:
            return int(np.atleast_1d(itp)[r])
    return 2


def _tabdist(dist):
    """Return the object exposing .energy (incident) + .energy_out (Tabulars), or None."""
    if hasattr(dist, 'energy_out') and hasattr(dist, 'energy') and not isinstance(getattr(dist, 'energy'), openmc.data.LevelInelastic):
        return dist
    ed = getattr(dist, 'energy', None)
    if ed is not None and hasattr(ed, 'energy_out') and hasattr(ed, 'energy'):
        return ed
    return None


def _unitbase_anchors(td, edges, per=4):
    """Precompute unit-base group-fraction at sub-sampled incident energies -> (E_anchor, GF[n,G])."""
    ein = np.asarray(td.energy, float); eos = td.energy_out; K = len(eos)
    Ea, GF = [], []
    for k in range(K - 1):
        hist = _incident_code(td, k) == 1
        for s in range(per):
            f = s / per
            E = ein[k] + f * (ein[k + 1] - ein[k])
            gf = _tab_frac(eos[k], edges) if hist else _unitbase_frac(eos[k], eos[k + 1], f, edges)
            if gf is None:
                gf = np.zeros(len(edges) - 1)
            Ea.append(E); GF.append(gf)
    gf = _tab_frac(eos[-1], edges)
    Ea.append(ein[-1]); GF.append(gf if gf is not None else np.zeros(len(edges) - 1))
    return np.array(Ea), np.array(GF)


def _freegas_gf(E, A, kT, edges, nv=28, nmu=14):
    """Free-gas (ideal-gas) elastic energy-transfer group fractions at incident energy E.
    Quadrature over the target Maxwellian (speed v_t, cosine mu_t); for each target the
    isotropic-CM scatter gives E' uniform over [0.5(Vcm-w)^2, 0.5(Vcm+w)^2]. Captures
    thermal up-scatter and broadening (reduces to Wigner-Wilkins for A=1). Units: E,kT in
    eV, speeds in sqrt(eV) with m_n=1."""
    vn = np.sqrt(2.0 * E); vth = np.sqrt(2.0 * kT / max(A, 1e-9))
    vt = np.linspace(0.02 * vth, 6.0 * vth, nv)
    wv = vt**2 * np.exp(-A * vt**2 / (2.0 * kT))                 # Maxwellian speed weight
    mu = np.linspace(-1.0, 1.0, nmu)
    VT = vt[:, None]; MU = mu[None, :]
    vrel = np.sqrt(np.clip(vn*vn + VT*VT - 2*vn*VT*MU, 0, None))
    Vcm = np.sqrt(np.clip(vn*vn + A*A*VT*VT + 2*A*vn*VT*MU, 0, None)) / (A + 1.0)
    w = (A / (A + 1.0)) * vrel
    Elo = (0.5 * (Vcm - w)**2).ravel(); Ehi = (0.5 * (Vcm + w)**2).ravel()
    wgt = ((wv[:, None]) * np.ones_like(mu)[None, :] * vrel).ravel()   # rate ~ Maxwell * v_rel
    rng = np.clip(Ehi - Elo, 1e-30, None); tot = wgt.sum()
    if tot <= 0:
        return None
    cdf = (np.clip((edges[:, None] - Elo[None, :]) / rng[None, :], 0, 1) * wgt[None, :]).sum(1) / tot
    return np.diff(cdf)


def scatter_matrix(material, groups, temperature=294.0, cross_sections=None, source=None,
                   wmode='nr', return_p1=False, thermal=True, phi_ext=None, grid_ext=None):
    groups = _as_energy_groups(groups)
    edges = np.asarray(groups.group_edges, float)
    emin, emax = edges[0], edges[-1]; G = groups.num_groups
    if cross_sections is None:
        cross_sections = openmc.config['cross_sections']
    datalib = openmc.data.DataLibrary.from_xml(cross_sections)
    dens = material.get_nuclide_atom_densities()
    incs, temp_str, grids = {}, {}, [edges]
    for nuc in dens:
        e = datalib.get_by_material(nuc, data_type='neutron')
        inc = openmc.data.IncidentNeutron.from_hdf5(e['path']); incs[nuc] = inc
        ts = _nearest_temperature(inc, temperature); temp_str[nuc] = ts
        grids.append(np.asarray(inc.energy[ts]))
    grid = np.unique(np.concatenate(grids)); grid = grid[(grid >= emin) & (grid <= emax)]
    # panel sub-grid: guarantee interior points in every group so the incoming
    # integration (esp. the elastic in-group vs down-scatter split) is resolved even
    # where the pointwise data is sparse (smooth fast range). Without this, a group
    # holding only its lower edge sends all elastic to the group below (diagonal collapse).
    sub = np.concatenate([np.geomspace(edges[g], edges[g + 1], 17)[1:-1] for g in range(G)])
    grid = np.unique(np.concatenate([grid, sub]))
    if grid_ext is not None:                            # use a caller-supplied grid (e.g. transport grid)
        grid = np.asarray(grid_ext, float); grid = grid[(grid >= emin) & (grid <= emax)]

    sigma_t = _macroscopic(incs, dens, temp_str, grid, 1)
    sigma_t = sigma_t + _apply_urr(incs, dens, temp_str, grid, sigma_t, temperature)[1]
    w = 1.0 / np.clip(grid, 1e-11, None)
    if source is not None:
        w = w + _source_pdf(source, grid)
    if wmode == 'sd':                                  # slowing-down weighting flux (coarse, old)
        from sd_flux import _slowing_down_weight
        phi = _slowing_down_weight(incs, dens, temp_str, grid, sigma_t, source)
    elif wmode == 'sdp':                               # PROPER fine-grid slowing-down flux
        from sd_flux import _slowing_down_flux_fine
        phi = _slowing_down_flux_fine(incs, dens, temp_str, grid, sigma_t, source)
    elif wmode == 'ir':                                # intermediate-resonance flux
        removed = np.zeros_like(grid)
        for nuc, n in dens.items():
            inc2 = incs[nuc]; A2 = inc2.atomic_weight_ratio; lam = 4.0 * A2 / (A2 + 1.0) ** 2
            if lam >= 1.0:
                continue
            ts2 = temp_str[nuc]; sti = inc2[1].xs[ts2](grid)
            try:
                sai = inc2[101].xs[ts2](grid)
            except KeyError:
                try:
                    sai = inc2[102].xs[ts2](grid)
                except KeyError:
                    sai = np.zeros_like(grid)
            removed += (1.0 - lam) * n * np.clip(sti - sai, 0.0, None)
        phi = w / np.clip(sigma_t - removed, 1e-30, None)
    else:                                              # 'nr' narrow-resonance (default)
        phi = w / np.clip(sigma_t, 1e-30, None)

    if phi_ext is not None:                            # external (e.g. deterministic-transport) local flux
        eg, ev = phi_ext
        phi = np.exp(np.interp(np.log(grid), np.log(eg), np.log(np.clip(ev, 1e-300, None))))

    dwid = np.empty_like(grid)
    dwid[1:-1] = 0.5 * (grid[2:] - grid[:-2]); dwid[0] = 0.5*(grid[1]-grid[0]); dwid[-1] = 0.5*(grid[-1]-grid[-2])
    wt = phi * dwid
    gi = np.clip(np.searchsorted(edges, grid, side='right') - 1, 0, G - 1)
    M = np.zeros((G, G)); denom = np.zeros(G); np.add.at(denom, gi, wt)
    M1 = np.zeros((G, G)) if return_p1 else None             # P1 (mu_lab-weighted) outscatter, elastic

    def overlap_frac(lo, hi):                                # uniform deposit over [lo,hi] -> len-G
        if hi <= lo:
            return None
        ov = np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0, None)
        return ov / (hi - lo)

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
            nz = np.nonzero(xs > 0)[0]
            if nz.size == 0:
                continue
            src = n * xs * wt
            if is_el:                                        # ---- elastic: CM angular distribution ----
                try:
                    ang = r.products[0].distribution[0].angle; aE = np.asarray(ang.energy)
                except Exception:
                    ang = None
                # free-gas thermal kernel for light nuclides (target motion -> up-scatter)
                kT = 8.617e-5 * temperature; E_TH = 400.0 * kT   # match OpenMC free_gas_threshold
                do_fg = thermal and A <= 20.0
                tgf = {}
                if do_fg:
                    mid = np.sqrt(edges[:-1] * edges[1:])
                    for g in np.where(mid < E_TH)[0]:
                        gf = _freegas_gf(mid[g], A, kT, edges)
                        if gf is not None:
                            tgf[int(g)] = gf
                for i in nz:
                    E = grid[i]
                    if do_fg and E < E_TH and gi[i] in tgf:   # thermal: free-gas energy transfer
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
                    # the wide H down-scatter (mu~-1 -> E_out~0) into the lowest group.
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
            # ---- inelastic: loop all neutron products & their distributions ----
            for prod in r.products:
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
                    if isinstance(ed, openmc.data.LevelInelastic):          # discrete level
                        thr, mr = float(ed.threshold), float(ed.mass_ratio)
                        for i in nz:
                            E = grid[i]
                            if E <= thr:
                                continue
                            Ecm = mr * (E - thr)
                            if Ecm <= 0:
                                continue
                            base = Ecm + E / a1; amp = 2.0 * np.sqrt(E * Ecm) / (A + 1.0)
                            of = overlap_frac(base - amp, base + amp)
                            wgt = src[i] * yld[i] * app[i]
                            if of is None:
                                go = min(max(np.searchsorted(edges, base) - 1, 0), G - 1)
                                M[gi[i], go] += wgt
                            else:
                                M[gi[i]] += wgt * of
                        continue
                    td = _tabdist(dist)                                     # continuum / (n,xn)
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
                            fr = (E - Ea[j-1]) / (Ea[j] - Ea[j-1])
                            gf = (1 - fr) * GF[j-1] + fr * GF[j]
                        M[gi[i]] += src[i] * yld[i] * app[i] * gf

    # fold upscatter artifacts (E_out>E_in; ascending go>gi) into the in-group element,
    # but KEEP real thermal up-scatter from the free-gas kernel (below E_TH)
    _midf = np.sqrt(edges[:-1] * edges[1:]); _eth = 400.0 * 8.617e-5 * temperature
    for g in range(G - 1):
        if thermal and _midf[g] < _eth:
            continue
        up = M[g, g+1:].sum()
        if up:
            M[g, g] += up; M[g, g+1:] = 0.0

    M = M / np.clip(denom[:, None], 1e-30, None)
    if return_p1:
        sigma_s1 = (M1.sum(1) / np.clip(denom, 1e-30, None))[::-1]   # P1 outscatter moment per group
        return M[::-1, ::-1], sigma_s1
    return M[::-1, ::-1]                                    # -> OpenMC ordering (group1 = high E)


if __name__ == "__main__":
    full = np.asarray(openmc.mgxs.GROUP_STRUCTURES["VITAMIN-J-42"]); EDGES = full[full <= 2.0e7]
    G = openmc.mgxs.EnergyGroups(EDGES); SRC = openmc.stats.muir(e0=14.06e6, m_rat=5.0, kt=20000.0)
    fe = openmc.Material(name="Fe56"); fe.add_nuclide("Fe56", 1.0); fe.set_density("g/cm3", 7.87)
    lib = openmc.MGXSLibrary.from_hdf5("cache_VITAMIN-J-42_mw1.h5")
    xs = [x for x in lib.xsdatas if x.name.startswith("Fe56")][0]
    mc = np.array(xs._scatter_matrix[0])[..., 0]
    t0 = time.time(); det = scatter_matrix(fe, G, source=SRC)
    print(f"Fe56 det in {time.time()-t0:.1f}s  nonneg={bool(np.all(det>=-1e-9))} diag={np.trace(det):.3f} "
          f"upper={np.triu(det,1).sum():.3f} lower={np.tril(det,-1).sum():.3f}")
    sig = mc > 1e-3
    print(f"row-sum mean|Δ| {100*np.mean(np.abs(det.sum(1)[mc.sum(1)>1e-3]-mc.sum(1)[mc.sum(1)>1e-3])/mc.sum(1)[mc.sum(1)>1e-3]):.2f}%  "
          f"full {100*np.mean(np.abs(det[sig]-mc[sig])/mc[sig]):.2f}%")
