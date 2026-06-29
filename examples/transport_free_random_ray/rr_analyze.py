"""Consistent volume-averaged comparison of RR(slab) and RR(transport_free) vs the CE
reference, reading the three statepoints directly. CE flux is volume-INTEGRATED
(track-length); RR volume_normalized flux is volume-AVERAGED. Convert both to a common
volume-AVERAGED region spectrum using analytic shell volumes, then score."""
import sys, glob, numpy as np, openmc
# normalize argv so rr_fusion_bench's import-time group parsing sees the group name
_grp = sys.argv[1] if len(sys.argv) > 1 else "VITAMIN-J-42"
sys.argv = [sys.argv[0], "ce", _grp]
import rr_fusion_bench as B

EDGES, NG, LAYOUT = B.EDGES, B.NG, B.LAYOUT
GROUPS = B.GROUPS

# rebuild geometry to recover per-cell radii/volumes and the region->cells map
def cell_geometry():
    r_prev = 0.0; cells = []   # list of (region, r_in, r_out)
    for name, r_out, nsub in LAYOUT:
        rs = np.linspace(r_prev, r_out, nsub + 1)
        for k in range(nsub):
            cells.append((name, rs[k], rs[k + 1]))
        r_prev = r_out
    vols = np.array([4/3*np.pi*(ro**3 - ri**3) for _, ri, ro in cells])
    regions = [c[0] for c in cells]
    return regions, vols

def per_cell_flux(sp_path):
    sp = openmc.StatePoint(sp_path)
    t = sp.get_tally(name="flux_spectrum")
    return t.get_values(scores=["flux"]).reshape(-1, NG)  # [cell, group] low->high E

def region_avg(flux_cell, regions, vols, integrated):
    """Volume-averaged region spectrum [NG], OpenMC ordering (group1=high E).
    integrated=True: flux_cell is vol-integrated (CE) -> sum/V. False: vol-averaged (RR) -> V-weighted mean."""
    names = [n for n, _, _ in LAYOUT]
    out = {}
    for nm in names:
        idx = [i for i, r in enumerate(regions) if r == nm]
        V = vols[idx].sum()
        if integrated:
            spec = flux_cell[idx].sum(0) / V
        else:
            spec = (flux_cell[idx] * vols[idx][:, None]).sum(0) / V
        out[nm] = spec[::-1]
    return out

def shape_err(a, b):
    a = a / a.sum(); b = b / b.sum(); k = b > b.max() * 1e-6
    return 100 * np.mean(np.abs((a - b)[k] / b[k]))

regions, vols = cell_geometry()
ce_sp = sorted(glob.glob("run_ce/statepoint.*.h5"))[-1]
sl_sp = sorted(glob.glob("run_rr_stochastic_slab/statepoint.*.h5"))[-1]
tf_sp = sorted(glob.glob("run_rr_transport_free/statepoint.*.h5"))[-1]
ce = region_avg(per_cell_flux(ce_sp), regions, vols, integrated=True)
sl = region_avg(per_cell_flux(sl_sp), regions, vols, integrated=False)
tf = region_avg(per_cell_flux(tf_sp), regions, vols, integrated=False)

names = [n for n, _, _ in LAYOUT]
fw = "tungsten"
print(f"=== RR-vs-CE fusion shield, {GROUPS} ({NG} groups), correction=None ===")
print(f"statepoints: CE={ce_sp}  slab={sl_sp}  tf={tf_sp}\n")
print(f"{'region':9}| spectrum-shape err vs CE (%)  | attenuation phi_region/phi_tungsten")
print(f"{'':9}|   slab        tf   (winner)   | CE         slab       tf")
slab_wins = tf_wins = 0
for nm in names:
    if nm == "plasma":
        continue
    se_sl, se_tf = shape_err(sl[nm], ce[nm]), shape_err(tf[nm], ce[nm])
    win = "tf" if se_tf < se_sl else "slab"
    if nm in ("steel", "lithium", "concrete"):  # deep regions (downstream of resonant metals)
        tf_wins += se_tf < se_sl; slab_wins += se_tf >= se_sl
    a_ce = ce[nm].sum() / ce[fw].sum()
    a_sl = sl[nm].sum() / sl[fw].sum()
    a_tf = tf[nm].sum() / tf[fw].sum()
    print(f"{nm:9}|  {se_sl:6.2f}   {se_tf:6.2f}   ({win:4}) | {a_ce:.3e}  {a_sl:.3e}  {a_tf:.3e}")

# headline: attenuation accuracy (deep flux suppression vs CE) + deep spectrum
def atten_err(m):
    return 100 * np.mean([abs((m[nm].sum()/m[fw].sum()) / (ce[nm].sum()/ce[fw].sum()) - 1)
                          for nm in ("steel", "lithium", "concrete")])
print(f"\nDeep-attenuation error vs CE (mean over steel/Li/concrete):  "
      f"slab {atten_err(sl):.1f}%   transport_free {atten_err(tf):.1f}%")
deep = "concrete"
print(f"DEEP ({deep}) spectrum-shape error vs CE:  slab {shape_err(sl[deep], ce[deep]):.2f}%   "
      f"tf {shape_err(tf[deep], ce[deep]):.2f}%")
print(f"\nDeep-region (steel/Li/concrete) spectrum-shape wins:  transport_free {tf_wins}/3, slab {slab_wins}/3")

# spectra plot: normalized flux/group, CE vs RR(slab) vs RR(transport_free)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    mid = np.sqrt(EDGES[:-1] * EDGES[1:])[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, nm in zip(axs, ("tungsten", "concrete")):
        for lab, d, c, ls, lw in [("CE ref", ce, "k", "-", 1.6),
                                   ("RR slab", sl, "tab:orange", "--", 1.2),
                                   ("RR transport_free", tf, "tab:blue", "--", 1.2)]:
            y = d[nm] / d[nm].sum()
            ax.step(mid, y, where="mid", label=lab, color=c, ls=ls, lw=lw)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{nm} region - normalized flux/group ({GROUPS})")
        ax.set_xlabel("E [eV]"); ax.set_ylabel("normalized flux")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"rr_spectra_{GROUPS}.png", dpi=130, bbox_inches="tight")
    print(f"saved rr_spectra_{GROUPS}.png")
except Exception as e:
    print("plot skipped:", e)
