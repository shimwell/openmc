"""Real-geometry deterministic transport: 1D SPHERICAL Sn through the actual reference
geometry (concentric shells, central point source) -> per-shell local flux -> collapse
each material. Matches the material_wise/slab reference geometry exactly. No Monte Carlo."""
import sys, time, numpy as np, openmc, openmc.data
from openmc.mgxs.transport_free import _macroscopic, _apply_urr, _source_pdf, _nearest_temperature
from scatter_det import scatter_matrix
from sphere_sn import solve_sphere
from mats import materials
_trapz = getattr(np, 'trapezoid', None) or np.trapz
GS = sys.argv[1] if len(sys.argv) > 1 else "CCFE-709"
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
grids = [edges] + [np.asarray(INC[n].energy[TS[n]]) for n in allnuc]
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
MD = {nm: mdat(nm) for nm in ORDER}
# ---- spherical mesh: shell radii (W: 0-6, then 4cm shells), subdivided ----
rad = [0.0, 6.0] + [6.0+4.0*i for i in range(1, len(ORDER))]   # outer radii: 0,6,10,...,42
redge = [0.0]; cmat = []
for j in range(len(ORDER)):
    r0, r1 = rad[j], rad[j+1]; nc = 6 if j == 0 else 4
    for c in range(nc): redge.append(r0+(r1-r0)*(c+1)/nc); cmat.append(ORDER[j])
redge = np.array(redge); ncell = len(cmat)
V = 4*np.pi/3*(redge[1:]**3 - redge[:-1]**3)
ST = np.array([MD[nm][0] for nm in cmat])
lay_sl = []; s = 0
for j in range(len(ORDER)):
    nc = 6 if j == 0 else 4; lay_sl.append((ORDER[j], slice(s, s+nc))); s += nc
mu, wmu = np.polynomial.legendre.leggauss(8); alpha = np.zeros(9)
for m in range(8): alpha[m+1] = alpha[m] - mu[m]*wmu[m]
print(f"[{GS}] grid {N}, {ncell} spherical cells (R={redge[-1]:.0f}cm)", flush=True)
# ---- energy sweep ----
phi = np.zeros((ncell, N)); t0 = time.time()
for i in range(N-1, -1, -1):
    Svol = np.zeros(ncell)
    for nm, sl in lay_sl:
        for cf, jh in MD[nm][1]:
            j = jh[i]
            if j > i+1: Svol[sl] += phi[sl, i+1:j] @ cf[i+1:j]
    Svol = Svol/np.clip(V, 1e-30, None)               # downscatter source per unit volume
    Svol[0] += inc_src[i]/V[0]                          # central point source in innermost cell
    phi[:, i] = solve_sphere(redge, ST[:, i], Svol, mu, wmu, alpha)
print(f"  spherical transport solve {time.time()-t0:.0f}s", flush=True)
# ---- collapse each material vs its shell flux (spherical volume average) ----
def coll(p, sig):
    out = np.zeros(G)
    for g in range(G):
        k = (grid >= edges[g]) & (grid <= edges[g+1])
        if k.sum() < 2: continue
        x, pp, sg = grid[k], p[k], sig[k]; dd = _trapz(pp, x); out[g] = _trapz(sg*pp, x)/dd if dd > 0 else 0
    return out[::-1]
def lib(tag, nm, kind):
    L = openmc.MGXSLibrary.from_hdf5(f"scatref_{GS}_{tag}.h5"); x = [a for a in L.xsdatas if a.name.startswith(nm)][0]
    return np.array(x._total[0]) if kind == 't' else np.array(x._scatter_matrix[0])[..., 0]
def et(a, b): k = np.abs(b) > 1e-9; return 100*np.mean(np.abs((a-b)[k]/b[k]))
def rs(M, Nr): r = Nr.sum(1) > 1e-3; return 100*np.mean(np.abs((M.sum(1)-Nr.sum(1))[r]/Nr.sum(1)[r]))
def mlg(M): r = M.sum(1); return np.where(r > 0, (M@Ulg)/np.clip(r, 1e-30, None), 0)
def shp(M, Nr): a, b = mlg(M), mlg(Nr); r = Nr.sum(1) > 1e-3; return float(np.mean(np.abs((a-b)[r])))
print(f"{'material':9}| TOTAL: NR / TR / slab | ROWSUM: NR / TR / slab | SHAPE: TR / slab", flush=True)
for nm, sl in lay_sl:
    Vt = V[sl]; phi_loc = (phi[sl]*Vt[:, None]).sum(0)/Vt.sum(); st_fe = MD[nm][0]
    phi_nr = (1.0/np.clip(grid, 1e-11, None)+inc_src)/np.clip(st_fe, 1e-30, None)
    Msh = scatter_matrix(mats[nm], GE, source=SRC)
    d_ = mats[nm].get_nuclide_atom_densities()
    sab = _macroscopic({k: INC[k] for k in d_}, d_, TS, grid, 101)
    if sab is None: sab = _macroscopic({k: INC[k] for k in d_}, d_, TS, grid, 102)
    sscat = np.clip(st_fe-(sab if sab is not None else 0.0), 0, None)
    Mtr = Msh*(coll(phi_loc, sscat)/np.clip(Msh.sum(1), 1e-30, None))[:, None]
    mwt = lib("mw", nm, 't'); slt = lib("slab", nm, 't'); mws = lib("mw", nm, 's'); sls = lib("slab", nm, 's')
    print(f"{nm:9}|  {et(coll(phi_nr,st_fe),mwt):4.2f} {et(coll(phi_loc,st_fe),mwt):4.2f} {et(slt,mwt):4.2f} | {rs(Msh,mws):4.2f} {rs(Mtr,mws):4.2f} {rs(sls,mws):4.2f} | {shp(Mtr,mws):.3f} {shp(sls,mws):.3f}", flush=True)
