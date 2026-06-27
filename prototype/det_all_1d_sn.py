"""Deterministic 1D Sn transport collapse for ALL test materials vs material_wise + slab.
Total XS, scatter row-sum, scatter shape. O(N) sliding-window down-scatter (fast for H)."""
import sys, time, numpy as np, openmc, openmc.data
from openmc.mgxs.transport_free import _macroscopic, _apply_urr, _source_pdf, _nearest_temperature
from scatter_det import scatter_matrix
from mats import materials
_trapz = getattr(np, 'trapezoid', None) or np.trapz
GS = sys.argv[1] if len(sys.argv) > 1 else "CCFE-709"
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
SRC = openmc.stats.muir(e0=14.06e6, m_rat=5.0, kt=20000.0)
datalib = openmc.data.DataLibrary.from_xml(openmc.config['cross_sections'])
_ms = materials(); mats = {m.name: m for m in _ms}; ORDER = [m.name for m in _ms]
E = np.asarray(openmc.mgxs.GROUP_STRUCTURES[GS]); E = E[E <= 2e7]; edges = E; G = len(E)-1
GE = openmc.mgxs.EnergyGroups(E); mid = np.sqrt(edges[:-1]*edges[1:])[::-1]; Ulg = -np.log(mid)
allnuc = set()
for m in _ms: allnuc |= set(m.get_nuclide_atom_densities())
INC = {}; TS = {}
for nuc in allnuc:
    inc = openmc.data.IncidentNeutron.from_hdf5(datalib.get_by_material(nuc, data_type='neutron')['path'])
    INC[nuc] = inc; TS[nuc] = _nearest_temperature(inc, 294.0)
mu, wmu = np.polynomial.legendre.leggauss(8); pos = mu > 0; neg = mu < 0; mu_p = mu[pos]; mu_n = np.abs(mu[neg])

def lib(tag, nm, kind):
    L = openmc.MGXSLibrary.from_hdf5(f"scatref_{GS}_{tag}.h5")
    x = [a for a in L.xsdatas if a.name.startswith(nm)][0]
    return np.array(x._total[0]) if kind == 't' else np.array(x._scatter_matrix[0])[..., 0]

def run(TARGET):
    LAY = [(ORDER[j], 6.0 if j == 0 else 4.0, 6 if j == 0 else 4) for j in range(ORDER.index(TARGET)+1)]
    matn = [l[0] for l in LAY]; pn = set()
    for nm in matn: pn |= set(mats[nm].get_nuclide_atom_densities())
    grids = [edges] + [np.asarray(INC[n].energy[TS[n]]) for n in pn]
    grid = np.unique(np.concatenate(grids)); grid = grid[(grid >= edges[0]) & (grid <= edges[-1])]
    grid = np.unique(np.concatenate([grid, np.concatenate([np.geomspace(edges[g], edges[g+1], 9)[1:-1] for g in range(G)])]))
    u = np.log(grid); keep = [0]; last = u[0]
    for j in range(1, len(u)):
        if u[j]-last >= 4e-4 or j == len(u)-1: keep.append(j); last = u[j]
    grid = grid[keep]; N = len(grid)
    dE = np.empty(N); dE[1:-1] = 0.5*(grid[2:]-grid[:-2]); dE[0] = 0.5*(grid[1]-grid[0]); dE[-1] = 0.5*(grid[-1]-grid[-2])
    inc_src = _source_pdf(SRC, grid)
    def mdat(nm):
        d = mats[nm].get_nuclide_atom_densities()
        st = _macroscopic({k: INC[k] for k in d}, d, TS, grid, 1); st = st + _apply_urr({k: INC[k] for k in d}, d, TS, grid, st, 294.0)[1]
        kn = []
        for nuc, n in d.items():
            A = INC[nuc].atomic_weight_ratio; a = ((A-1)/(A+1))**2
            if a >= 1-1e-9: continue
            try: ss = n*INC[nuc][2].xs[TS[nuc]](grid)
            except Exception: continue
            if not np.any(ss > 0): continue
            kn.append((ss/((1-a)*np.clip(grid, 1e-30, None))*dE, np.searchsorted(grid, grid/a, side='right')))
        return st, kn
    MD = {nm: mdat(nm) for nm in matn}
    cells = []
    for nm, th, nc in LAY:
        for _ in range(nc): cells.append((nm, th/nc))
    ncell = len(cells); dx = np.array([c[1] for c in cells]); cmat = [c[0] for c in cells]
    ST = np.array([MD[nm][0] for nm in cmat])
    lay_sl = []; s = 0
    for nm, th, nc in LAY: lay_sl.append((nm, slice(s, s+nc))); s += nc
    phi = np.zeros((ncell, N))
    for i in range(N-1, -1, -1):
        Q = np.zeros(ncell)                                    # isotropic elastic down-scatter source
        for nm, sl in lay_sl:
            for cf, jh in MD[nm][1]:
                j = jh[i]
                if j > i+1: Q[sl] += phi[sl, i+1:j] @ cf[i+1:j]
        sigt = ST[:, i]; pa = np.zeros((ncell, 8))
        psn = np.full(mu_p.size, inc_src[i])
        for c in range(ncell):
            tM = 2*mu_p/dx[c]; po = np.clip((Q[c]+psn*(tM-sigt[c]))/(tM+sigt[c]), 0, None); pa[c, pos] = 0.5*(psn+po); psn = po
        psn = np.zeros(mu_n.size)
        for c in range(ncell-1, -1, -1):
            tM = 2*mu_n/dx[c]; po = np.clip((Q[c]+psn*(tM-sigt[c]))/(tM+sigt[c]), 0, None); pa[c, neg] = 0.5*(psn+po); psn = po
        phi[:, i] = pa @ wmu
    ti = [c for c in range(ncell) if cmat[c] == TARGET]; Vt = dx[ti]
    phi_loc = (phi[ti]*Vt[:, None]).sum(0)/Vt.sum(); st_fe = MD[TARGET][0]
    def coll(p, sig):
        out = np.zeros(G)
        for g in range(G):
            k = (grid >= edges[g]) & (grid <= edges[g+1])
            if k.sum() < 2: continue
            x, pp, sg = grid[k], p[k], sig[k]; dd = _trapz(pp, x); out[g] = _trapz(sg*pp, x)/dd if dd > 0 else 0
        return out[::-1]
    phi_nr = (1.0/np.clip(grid, 1e-11, None)+inc_src)/np.clip(st_fe, 1e-30, None)
    Msh = scatter_matrix(mats[TARGET], GE, source=SRC)
    d_ = mats[TARGET].get_nuclide_atom_densities()
    sab = _macroscopic({k: INC[k] for k in d_}, d_, TS, grid, 101)
    if sab is None: sab = _macroscopic({k: INC[k] for k in d_}, d_, TS, grid, 102)
    sscat = np.clip(st_fe-(sab if sab is not None else 0.0), 0, None)
    Mtr = Msh*(coll(phi_loc, sscat)/np.clip(Msh.sum(1), 1e-30, None))[:, None]
    return coll(phi_nr, st_fe), coll(phi_loc, st_fe), Msh, Mtr, N

def et(a, b): k = np.abs(b) > 1e-9; return 100*np.mean(np.abs((a-b)[k]/b[k]))
def rs(M, Nr): r = Nr.sum(1) > 1e-3; return 100*np.mean(np.abs((M.sum(1)-Nr.sum(1))[r]/Nr.sum(1)[r]))
def mlg(M): r = M.sum(1); return np.where(r > 0, (M@Ulg)/np.clip(r, 1e-30, None), 0)
def shp(M, Nr): a, b = mlg(M), mlg(Nr); r = Nr.sum(1) > 1e-3; return float(np.mean(np.abs((a-b)[r])))

print(f"[{GS}]  TOTAL%: NR / TRANSPORT / slab  |  ROWSUM%: NR / TR / slab  |  SHAPE: TR / slab", flush=True)
for nm in (ORDER if ONLY is None else [ONLY]):
    t0 = time.time()
    try:
        tnr, ttr, Msh, Mtr, N = run(nm)
        mwt = lib("mw", nm, 't'); slt = lib("slab", nm, 't'); mws = lib("mw", nm, 's'); sls = lib("slab", nm, 's')
        win = "WIN" if (et(ttr, mwt) <= et(slt, mwt) and rs(Mtr, mws) <= rs(sls, mws) and shp(Mtr, mws) <= shp(sls, mws)*1.15) else ""
        print(f"{nm:9}| {et(tnr,mwt):5.2f} {et(ttr,mwt):5.2f} {et(slt,mwt):5.2f} | {rs(Msh,mws):5.2f} {rs(Mtr,mws):5.2f} {rs(sls,mws):5.2f} | {shp(Mtr,mws):.3f} {shp(sls,mws):.3f}  {win} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"{nm:9}| ERROR {type(e).__name__}: {e}", flush=True)
