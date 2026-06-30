"""Inelastic + (n,2n)/(n,3n) down-scatter source for the deterministic transport.
Builds a COARSE transfer matrix M[out,in] (macroscopic, incl. multiplicity) per material
-- inelastic outgoing is smooth so a coarse grid + interpolation is accurate -- and applies
it to the (fine) flux to give the inelastic slowing-down source, added to the elastic source
in an outer iteration."""
import numpy as np
from sd_flux import _outgoing

def coarse_grid(emin, emax, per_decade=50):
    nb = int(max(per_decade*np.log10(emax/emin), 40))
    be = np.logspace(np.log10(emin), np.log10(emax), nb+1)
    return np.sqrt(be[:-1]*be[1:]), be                          # centers, edges

def build_inel_transfer(incs, dens, ts, cg, be):
    """M[out,in] (1/cm) = macroscopic inelastic+(n,xn) transfer rate per unit flux, incl mult."""
    nb = len(cg); M = np.zeros((nb, nb))
    for nuc, dn in dens.items():
        inc = incs[nuc]; t = ts[nuc]
        for mt, r in inc.reactions.items():
            if not ((51 <= mt <= 91) or mt in (16, 17)): continue
            try: sig = dn*r.xs[t](cg)
            except Exception: continue
            if not np.any(sig > 0): continue
            prods = [p for p in r.products if p.particle == 'neutron']
            if not prods: continue
            try:
                mult = np.atleast_1d(np.asarray(prods[0].yield_(cg), float))
                if mult.size == 1: mult = np.full(nb, float(mult[0]))
            except Exception:
                mult = np.ones(nb)
            try: og = _outgoing(prods[0].distribution[0])
            except Exception: og = None
            if og is None: continue
            if og[0] == 'delta':                                # discrete level
                thr, mr = og[1], og[2]; eout = mr*(cg-thr)
                for k in range(nb):
                    if sig[k] <= 0 or cg[k] <= thr or eout[k] < be[0]: continue
                    l = min(max(np.searchsorted(be, eout[k])-1, 0), nb-1)
                    M[l, k] += sig[k]*mult[k]
            else:                                               # 'tab' continuum / (n,xn)
                ein, eos = og[1], og[2]
                for k in range(nb):
                    if sig[k] <= 0 or cg[k] < ein[0]: continue
                    tb = eos[min(np.searchsorted(ein, cg[k]), len(eos)-1)]
                    cx, cp = np.asarray(tb.x, float), np.asarray(tb.p, float)
                    cc = np.zeros_like(cx); cc[1:] = np.cumsum(0.5*(cp[1:]+cp[:-1])*np.diff(cx))
                    if cc[-1] <= 0: continue
                    Wb = np.diff(np.interp(be, cx, cc/cc[-1], left=0.0, right=1.0))
                    M[:, k] += sig[k]*mult[k]*Wb
    return M

def inel_source_fine(M, phi_fine, grid, dE, cg, be):
    """Given coarse transfer M and a fine flux phi_fine[N], return the inelastic source
    density on the fine grid (same units as the elastic down-scatter source)."""
    binidx = np.clip(np.searchsorted(be, grid)-1, 0, len(cg)-1)
    Phi = np.zeros(len(cg)); np.add.at(Phi, binidx, phi_fine*dE)   # bin-integrated flux
    src_bin = M @ Phi                                              # total rate into each out bin /vol
    width = np.diff(be)
    q_coarse = src_bin/np.clip(width, 1e-30, None)                 # rate density per bin
    return q_coarse[binidx]                                        # piecewise-constant onto fine grid
