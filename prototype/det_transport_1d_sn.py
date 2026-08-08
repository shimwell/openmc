"""Deterministic 1D Sn slowing-down transport to get a CROSS-TALK-aware weighting
flux, then collapse a material against its LOCAL flux. No Monte Carlo.

Physics: fixed-source 1D slab transport on the fine pointwise energy grid. Neutrons
enter at x=0 (14 MeV muir), slow down (isotropic-CM elastic kernel) and attenuate
through the layered shield; the flux reaching the deep material carries the upstream
resonance imprint (the inter-material cross-talk stochastic_slab captures). On a fine
energy POINT grid elastic scatter is strictly down in energy -> a single high->low
energy sweep, each energy a one-pass Sn spatial solve (no within-energy scatter).
"""
import sys, time, numpy as np, openmc, openmc.data
from openmc.mgxs.transport_free import _macroscopic, _apply_urr, _source_pdf, _nearest_temperature
from mats import materials
_trapz = getattr(np, 'trapezoid', None) or np.trapz

GS = sys.argv[1] if len(sys.argv) > 1 else "CCFE-709"
SRC = openmc.stats.muir(e0=14.06e6, m_rat=5.0, kt=20000.0)
datalib = openmc.data.DataLibrary.from_xml(openmc.config['cross_sections'])
_ms = materials(); mats = {m.name: m for m in _ms}; ORDER = [m.name for m in _ms]
# cross-talk path = all shells from the source up to and including the target (vacuum after)
TARGET = sys.argv[2] if len(sys.argv) > 2 else "Fe56"
LAYERS = [(ORDER[j], 6.0 if j == 0 else 4.0, 6 if j == 0 else 4) for j in range(ORDER.index(TARGET)+1)]
NMU = 8                                                    # S8 Gauss-Legendre

E = np.asarray(openmc.mgxs.GROUP_STRUCTURES[GS]); E = E[E <= 2e7]; edges = E; G = len(E)-1
GE = openmc.mgxs.EnergyGroups(E)

# ---- energy grid: union of all path materials' nuclide grids + per-group sub-grid ----
matnames = [l[0] for l in LAYERS]
allnuc = set()
for nm in matnames: allnuc |= set(mats[nm].get_nuclide_atom_densities())
incs={}; ts_={}; grids=[edges]
for nuc in allnuc:
    inc=openmc.data.IncidentNeutron.from_hdf5(datalib.get_by_material(nuc,data_type='neutron')['path'])
    incs[nuc]=inc; ts_[nuc]=_nearest_temperature(inc,294.0); grids.append(np.asarray(inc.energy[ts_[nuc]]))
grid=np.unique(np.concatenate(grids)); grid=grid[(grid>=edges[0])&(grid<=edges[-1])]
sub=np.concatenate([np.geomspace(edges[g],edges[g+1],9)[1:-1] for g in range(G)])
grid=np.unique(np.concatenate([grid,sub]))
# thin by min lethargy spacing (keeps resonances ~100 pts wide, caps redundant density)
u=np.log(grid); keep=[0]; last=u[0]
for j in range(1,len(u)):
    if u[j]-last>=4e-4 or j==len(u)-1: keep.append(j); last=u[j]
grid=grid[keep]; N=len(grid)
dE=np.empty(N); dE[1:-1]=0.5*(grid[2:]-grid[:-2]); dE[0]=0.5*(grid[1]-grid[0]); dE[-1]=0.5*(grid[-1]-grid[-2])
print(f"[{GS}] grid {N} pts, {sum(l[2] for l in LAYERS)} cells", flush=True)

# ---- per-material total + per-nuclide elastic slowing-down kernel coef ----
def mat_data(nm):
    d=mats[nm].get_nuclide_atom_densities()
    st=_macroscopic({k:incs[k] for k in d},d,ts_,grid,1)
    st=st+_apply_urr({k:incs[k] for k in d},d,ts_,grid,st,294.0)[1]
    kerns=[]                                               # (coefdE_j = n*sigma_s,el/((1-a)E) * dE,  jhi[i])
    for nuc,n in d.items():
        inc=incs[nuc]; ts=ts_[nuc]; A=inc.atomic_weight_ratio; a=((A-1)/(A+1))**2
        if a>=1-1e-9: continue
        try: ss=n*inc[2].xs[ts](grid)
        except Exception: continue
        if not np.any(ss>0): continue
        kerns.append((ss/((1-a)*np.clip(grid,1e-30,None))*dE, np.searchsorted(grid,grid/a,side='right')))
    return st, kerns
MD={nm:mat_data(nm) for nm in matnames}

# ---- cell geometry ----
cells=[]                                                   # (material, dx)
for nm,thick,nc in LAYERS:
    for _ in range(nc): cells.append((nm, thick/nc))
ncell=len(cells); dx=np.array([c[1] for c in cells])
cmat=[c[0] for c in cells]
ST=np.array([MD[nm][0] for nm in cmat])                    # [ncell, N] total xs per cell
mu,wmu=np.polynomial.legendre.leggauss(NMU)                # nodes/weights on [-1,1], sum w=2
pos=mu>0; neg=mu<0; mu_p=mu[pos]; mu_n=np.abs(mu[neg])
inc_src=_source_pdf(SRC,grid)                              # incident spectrum at x=0 (muir)
# contiguous cell block per layer (each layer = one material)
lay_sl=[]; s=0
for nm,thick,nc in LAYERS: lay_sl.append((nm,slice(s,s+nc))); s+=nc

# ---- energy sweep (high->low): one Sn spatial solve per energy ----
phi=np.zeros((ncell,N)); t0=time.time()
for i in range(N-1,-1,-1):
    # isotropic down-scatter source per cell (elastic, from already-solved higher energies)
    Q=np.zeros(ncell)
    for nm,sl in lay_sl:
        for coefdE,jhi in MD[nm][1]:
            jh=jhi[i]
            if jh>i+1: Q[sl]+=phi[sl,i+1:jh]@coefdE[i+1:jh]
    sigt=ST[:,i]
    psi_avg=np.zeros((ncell,NMU))
    psin=np.full(mu_p.size,inc_src[i])                     # forward: incident muir at x=0
    for c in range(ncell):
        tM=2*mu_p/dx[c]; den=tM+sigt[c]
        psout=np.clip((Q[c]+psin*(tM-sigt[c]))/den,0,None) # Q[c]=src*2 (src=Q/2)
        psi_avg[c,pos]=0.5*(psin+psout); psin=psout
    psin=np.zeros(mu_n.size)                                # backward: vacuum at x=L
    for c in range(ncell-1,-1,-1):
        tM=2*mu_n/dx[c]; den=tM+sigt[c]
        psout=np.clip((Q[c]+psin*(tM-sigt[c]))/den,0,None)
        psi_avg[c,neg]=0.5*(psin+psout); psin=psout
    phi[:,i]=psi_avg@wmu                                   # scalar flux = sum_n w_n psi_n
print(f"  transport solve {time.time()-t0:.1f}s ({N} energies)", flush=True)

# ---- collapse TARGET against its LOCAL (volume-avg) flux ----
ti=[c for c in range(ncell) if cmat[c]==TARGET]
Vfe=dx[ti]; phi_loc=(phi[ti]*Vfe[:,None]).sum(0)/Vfe.sum()   # vol-avg flux in target layer
st_fe=MD[TARGET][0]
def collapse(phi1d):
    out=np.zeros(G)
    for g in range(G):
        k=(grid>=edges[g])&(grid<=edges[g+1])
        if k.sum()<2: continue
        x,p,s=grid[k],phi1d[k],st_fe[k]; d=_trapz(p,x); out[g]=_trapz(s*p,x)/d if d>0 else 0
    return out[::-1]
def lib(tag):
    L=openmc.MGXSLibrary.from_hdf5(f"scatref_{GS}_{tag}.h5"); x=[a for a in L.xsdatas if a.name.startswith(TARGET)][0]; return np.array(x._total[0])
mwt=lib("mw"); slabt=lib("slab")
def err(a): k=np.abs(mwt)>1e-9; return 100*np.mean(np.abs((a-mwt)[k]/mwt[k]))
phi_nr=(1.0/np.clip(grid,1e-11,None)+inc_src)/np.clip(st_fe,1e-30,None)
print(f"\n{TARGET} TOTAL %err vs material_wise ({GS}):")
print(f"  det-NR (isolated)   {err(collapse(phi_nr)):.2f}")
print(f"  det-TRANSPORT (1D)  {err(collapse(phi_loc)):.2f}")
print(f"  stochastic_slab     {err(slabt):.2f}")


# ---- SCATTER MATRIX for RR: kernel SHAPE (weighting-insensitive) x transport-flux
#      scatter XS (direct collapse). The SCALE/AMPX recipe: 2D matrix shape renormalised
#      to the self-shielded scatter cross section. ----
from scatter_det import scatter_matrix
def p0(tag):
    L=openmc.MGXSLibrary.from_hdf5(f"scatref_{GS}_{tag}.h5")
    x=[a for a in L.xsdatas if a.name.startswith(TARGET)][0]; return np.array(x._scatter_matrix[0])[...,0]
mw_s=p0("mw"); slab_s=p0("slab")
mid=np.sqrt(edges[:-1]*edges[1:])[::-1]; Ulg=-np.log(mid)
def rs(M,Nref): r=Nref.sum(1)>1e-3; return 100*np.mean(np.abs((M.sum(1)-Nref.sum(1))[r]/Nref.sum(1)[r]))
def mlg(M): r=M.sum(1); return np.where(r>0,(M@Ulg)/np.clip(r,1e-30,None),0)
def shp(M,Nref): a,b=mlg(M),mlg(Nref); r=Nref.sum(1)>1e-3; return float(np.mean(np.abs((a-b)[r])))
def coll(phi1d,sig):                                          # direct group collapse on the transport grid
    out=np.zeros(G)
    for g in range(G):
        k=(grid>=edges[g])&(grid<=edges[g+1])
        if k.sum()<2: continue
        x,p,s=grid[k],phi1d[k],sig[k]; dd=_trapz(p,x); out[g]=_trapz(s*p,x)/dd if dd>0 else 0
    return out[::-1]
M_shape=scatter_matrix(mats[TARGET],GE,source=SRC)            # kernel shape (NR weighting; shape ~ weighting-free)
d_=mats[TARGET].get_nuclide_atom_densities()
sabs=_macroscopic({k:incs[k] for k in d_},d_,ts_,grid,101)
if sabs is None: sabs=_macroscopic({k:incs[k] for k in d_},d_,ts_,grid,102)
sscat=np.clip(st_fe-(sabs if sabs is not None else 0.0),0,None)   # scatter XS ~ total - absorption
M_tr=M_shape*(coll(phi_loc,sscat)/np.clip(M_shape.sum(1),1e-30,None))[:,None]   # rows -> transport scatter XS
print(f"\n{TARGET} SCATTER MATRIX ({GS}):")
print(f"  ROWSUM %err:  det-NR {rs(M_shape,mw_s):.2f}   det-TRANSPORT {rs(M_tr,mw_s):.2f}   slab {rs(slab_s,mw_s):.2f}")
print(f"  SHAPE leth :  det-NR {shp(M_shape,mw_s):.3f}  det-TRANSPORT {shp(M_tr,mw_s):.3f}  slab {shp(slab_s,mw_s):.3f}")
