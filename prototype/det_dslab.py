"""DETERMINISTIC stochastic_slab (Option A): reproduce OpenMC's slab representative
weighting without Monte Carlo. Reflective box of mixed materials + volumetric source
=> spatially-flat flux => 0-D infinite-medium slowing-down spectrum of the material MIX.
Each material X weighted by phi_X(E) = q_mix(E)/Sigma_t,X(E)  (mix slowing-down source
x X's own resonance self-shielding). Order-free; compare vs MC slab + material_wise."""
import sys, time, numpy as np, openmc, openmc.data
from openmc.mgxs.transport_free import _macroscopic, _apply_urr, _source_pdf, _nearest_temperature
from sd_flux import _slowing_down_flux_fine
from scatter_det import scatter_matrix
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
# ---- grid (all nuclides) ----
grids = [edges] + [np.asarray(INC[n].energy[TS[n]]) for n in allnuc]
grid = np.unique(np.concatenate(grids)); grid = grid[(grid >= edges[0]) & (grid <= edges[-1])]
grid = np.unique(np.concatenate([grid, np.concatenate([np.geomspace(edges[g], edges[g+1], 9)[1:-1] for g in range(G)])]))
u = np.log(grid); keep = [0]; last = u[0]
for j in range(1, len(u)):
    if u[j]-last >= 4e-4 or j == len(u)-1: keep.append(j); last = u[j]
grid = grid[keep]; N = len(grid)
def macro(dens, mt):
    return _macroscopic({k: INC[k] for k in dens}, dens, TS, grid, mt)
def total_ss(dens):
    st = macro(dens, 1); return st + _apply_urr({k: INC[k] for k in dens}, dens, TS, grid, st, 294.0)[1]

# ---- homogenized mix (equal volume fraction per material, as slab uses 100 cells each) ----
mixn = {}
for m in _ms:
    for nuc, n in m.get_nuclide_atom_densities().items():
        mixn[nuc] = mixn.get(nuc, 0.0) + n/len(_ms)
st_mix = total_ss(mixn)
t0 = time.time()
phi_mix = _slowing_down_flux_fine(INC, mixn, TS, grid, st_mix, SRC)   # 0-D infinite-medium mix flux
q_mix = phi_mix * st_mix                                              # mix slowing-down source (smooth, degraded)
print(f"[{GS}] grid {N}, mix slowing-down solve {time.time()-t0:.0f}s", flush=True)

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

print(f"{'material':9}| TOTAL: dslab vs(mw) | slab vs(mw) || ROWSUM: dslab / slab || SHAPE: dslab / slab", flush=True)
for nm in ORDER:
    d = mats[nm].get_nuclide_atom_densities(); st_X = total_ss(d)
    phi_X = q_mix/np.clip(st_X, 1e-30, None)                          # mix source x X's own self-shielding
    Msh = scatter_matrix(mats[nm], GE, source=SRC)
    sab = macro(d, 101);  sab = sab if sab is not None else macro(d, 102)
    sscat = np.clip(st_X-(sab if sab is not None else 0.0), 0, None)
    Mds = Msh*(coll(phi_X, sscat)/np.clip(Msh.sum(1), 1e-30, None))[:, None]
    mwt = lib("mw", nm, 't'); slt = lib("slab", nm, 't'); mws = lib("mw", nm, 's'); sls = lib("slab", nm, 's')
    # vs mw (truth) AND vs slab (are we reproducing it?)
    print(f"{nm:9}|   {et(coll(phi_X,st_X),mwt):4.2f}  {et(slt,mwt):4.2f}  ||  {rs(Mds,mws):4.2f} {rs(sls,mws):4.2f}  ||  {shp(Mds,mws):.3f} {shp(sls,mws):.3f}   [dslab-vs-slab tot {et(coll(phi_X,st_X),slt):.2f}]", flush=True)
