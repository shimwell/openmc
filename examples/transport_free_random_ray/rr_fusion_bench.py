"""RR-vs-CE fusion-shield benchmark: does transport_free MGXS beat stochastic_slab when
fed to OpenMC's random ray solver, judged against a continuous-energy reference?

Geometry: spherical fusion shield using the neutronics-workshop tokamak materials/thicknesses
(14.1 MeV plasma source -> tungsten armor -> steel structure -> lithium blanket -> concrete
bioshield). Deep-penetration: the deep-region spectrum depends on the upstream metals' MGXS.

Metrics (normalization-independent, so CE-vs-RR is apples-to-apples):
  - deep-region group flux SHAPE (each spectrum normalized to unit sum)
  - radial attenuation profile (region flux / first-wall flux)

usage: rr_fusion_bench.py {ce | rr <method> | compare} [GROUPS]
"""
import sys, warnings, numpy as np, openmc
warnings.simplefilter("ignore")

_gpos = 3 if (len(sys.argv) > 1 and sys.argv[1] == "rr") else 2
GROUPS = sys.argv[_gpos] if len(sys.argv) > _gpos else "VITAMIN-J-42"
_full = np.asarray(openmc.mgxs.GROUP_STRUCTURES[GROUPS]); EDGES = _full[_full <= 2.0e7]
GE = openmc.mgxs.EnergyGroups(EDGES); NG = GE.num_groups
SRC_E = 14.1e6

# ---- materials (workshop compositions) -------------------------------------------------
def materials():
    plasma = openmc.Material(name="plasma")          # near-void DT-ish gas, source region
    plasma.add_element("H", 1.0); plasma.set_density("g/cm3", 1e-5)
    w = openmc.Material(name="tungsten"); w.add_element("W", 1.0); w.set_density("g/cm3", 19.3)
    steel = openmc.Material(name="steel")             # SS316-ish
    for el, f in [("Fe", 0.66), ("Cr", 0.17), ("Ni", 0.12), ("Mo", 0.025), ("Mn", 0.02)]:
        steel.add_element(el, f, "wo")
    steel.set_density("g/cm3", 7.93)
    li = openmc.Material(name="lithium"); li.add_element("Li", 1.0); li.set_density("g/cm3", 0.534)
    conc = openmc.Material(name="concrete")
    for el, f in [("O", 0.52), ("Si", 0.325), ("Ca", 0.06), ("Al", 0.033),
                  ("Fe", 0.014), ("H", 0.01), ("Na", 0.017), ("Mg", 0.002), ("K", 0.019)]:
        conc.add_element(el, f, "wo")
    conc.set_density("g/cm3", 2.3)
    return [plasma, w, steel, li, conc]

# region outer radii [cm]; sub = radial subdivisions (FSRs for RR spatial resolution)
LAYOUT = [("plasma", 30.0, 2), ("tungsten", 50.0, 3), ("steel", 70.0, 3),
          ("lithium", 120.0, 5), ("concrete", 170.0, 5)]

def build_model(energy_mode="continuous-energy"):
    mats = {m.name: m for m in materials()}
    surfs = []; r_prev = 0.0; cells = []; region_cells = {}
    for name, r_out, nsub in LAYOUT:
        rs = np.linspace(r_prev, r_out, nsub + 1)[1:]
        cl = []
        for k, r in enumerate(rs):
            s = openmc.Sphere(r=r)
            inner = -surfs[-1] if surfs else None
            cell = openmc.Cell(fill=mats[name], region=(-s & +surfs[-1]) if surfs else -s)
            surfs.append(s); cells.append(cell); cl.append(cell)
        region_cells[name] = cl
        r_prev = r_out
    surfs[-1].boundary_type = "vacuum"
    geom = openmc.Geometry(cells)
    st = openmc.Settings(); st.run_mode = "fixed source"; st.energy_mode = energy_mode
    st.batches = 100 if energy_mode == "continuous-energy" else 100
    st.particles = 200000 if energy_mode == "continuous-energy" else 2000
    # physical source: uniform over plasma region, 14.1 MeV (CE) / top group (MG)
    if energy_mode == "continuous-energy":
        e_dist = openmc.stats.Discrete([SRC_E], [1.0])
    else:
        mid = np.sqrt(EDGES[:-1] * EDGES[1:]); g = int(np.argmin(np.abs(mid - SRC_E)))
        e_dist = openmc.stats.Discrete([mid[g]], [1.0])
    space = openmc.stats.Point()  # placeholder; constrained below for RR
    st.source = openmc.IndependentSource(energy=e_dist, angle=openmc.stats.Isotropic())
    model = openmc.Model(geom, openmc.Materials(list(mats.values())), st)
    return model, region_cells

def add_tally(model, region_cells):
    """Flux per group in each region (volume-summed over its sub-cells)."""
    t = openmc.Tally(name="flux_spectrum")
    cf = openmc.CellFilter([c for cl in region_cells.values() for c in cl])
    ef = openmc.EnergyFilter(EDGES)
    t.filters = [cf, ef]; t.scores = ["flux"]
    model.tallies = openmc.Tallies([t])
    return [c.id for cl in region_cells.values() for c in cl]

def region_flux(sp_path, cell_ids, region_cells):
    """Return {region: group_flux[NG]} summing the sub-cells of each region."""
    sp = openmc.StatePoint(sp_path)
    t = sp.get_tally(name="flux_spectrum")
    fl = t.get_values(scores=["flux"]).reshape(len(cell_ids), NG)  # [cell, group] low->high E
    id_index = {cid: i for i, cid in enumerate(cell_ids)}
    out = {}
    for name, cl in region_cells.items():
        idx = [id_index[c.id] for c in cl]
        out[name] = fl[idx].sum(0)[::-1]  # -> high E = group 1 (OpenMC ordering)
    return out

# ---- stages ----------------------------------------------------------------------------
def run_ce():
    model, rc = build_model("continuous-energy")
    cell_ids = add_tally(model, rc)
    sp = model.run(cwd="run_ce")
    fl = region_flux(sp, cell_ids, rc)
    np.savez(f"rr_ref_ce_{GROUPS}.npz", cell_ids=cell_ids,
             **{k: v for k, v in fl.items()})
    print("CE reference done ->", f"rr_ref_ce_{GROUPS}.npz")
    for k, v in fl.items():
        print(f"  {k:9} integral flux = {v.sum():.4e}")

def _sanitize(path, floor=1e-5):
    """RR rejects zero/negative total XS. Clip total (and absorption) to a small floor
    where needed; applied identically to every method so the comparison stays fair.
    Returns the count of clipped (material, group) entries."""
    lib = openmc.MGXSLibrary.from_hdf5(path); n = 0
    for x in lib.xsdatas:
        tot = np.array(x._total[0])
        bad = tot <= 0
        if bad.any():
            n += int(bad.sum()); tot[bad] = floor
            x._total[0] = tot
            ab = np.array(x._absorption[0]); ab[bad] = np.maximum(ab[bad], 0.0); x._absorption[0] = ab
    if n:
        lib.export_to_hdf5(path)
    return n

def run_rr(method):
    model, rc = build_model("continuous-energy")          # start CE, then convert
    src_dist = model.settings.source[0].energy
    model.convert_to_multigroup(method=method, groups=GE,
                                mgxs_path=f"rr_mgxs_{method}_{GROUPS}.h5",
                                overwrite_mgxs_library=True, correction=None,
                                source_energy=openmc.stats.Discrete([SRC_E], [1.0]))
    nclip = _sanitize(f"rr_mgxs_{method}_{GROUPS}.h5")
    print(f"  [{method}] sanitized {nclip} zero/negative total-XS entries")
    # constrain the physical source to the plasma material (RR fixed-source domain)
    plasma_mat = [m for m in model.materials if m.name == "plasma"][0]
    model.settings.source = [openmc.IndependentSource(
        energy=src_dist, constraints={"domains": [plasma_mat]}, strength=1.0)]
    cell_ids = add_tally(model, rc)
    model.convert_to_random_ray()
    model.settings.random_ray["volume_normalized_flux_tallies"] = True
    model.settings.particles = 4000; model.settings.batches = 150; model.settings.inactive = 50
    sp = model.run(cwd=f"run_rr_{method}")
    fl = region_flux(sp, cell_ids, rc)
    np.savez(f"rr_res_{method}_{GROUPS}.npz", **{k: v for k, v in fl.items()})
    print(f"RR ({method}) done ->", f"rr_res_{method}_{GROUPS}.npz")
    for k, v in fl.items():
        print(f"  {k:9} integral flux = {v.sum():.4e}")

def compare():
    ref = np.load(f"rr_ref_ce_{GROUPS}.npz")
    regions = [n for n, _, _ in LAYOUT]
    methods = ["stochastic_slab", "transport_free"]
    res = {m: np.load(f"rr_res_{m}_{GROUPS}.npz") for m in methods}
    def shape_err(a, b):  # normalized-spectrum mean abs % error
        a = a / a.sum(); b = b / b.sum(); k = b > b.max() * 1e-6
        return 100 * np.mean(np.abs((a - b)[k] / b[k]))
    print(f"\n=== RR-vs-CE, {GROUPS} ({NG} groups) ===")
    print(f"{'region':9}| spectrum shape err (%) vs CE   | atten ratio (region/tungsten) CE / slab / tf")
    fw = "tungsten"
    for nm in regions:
        if nm == "plasma": continue
        ce = ref[nm]
        line = f"{nm:9}|  slab {shape_err(res['stochastic_slab'][nm], ce):6.2f}   tf {shape_err(res['transport_free'][nm], ce):6.2f}   |"
        a_ce = ce.sum() / ref[fw].sum()
        a_sl = res['stochastic_slab'][nm].sum() / res['stochastic_slab'][fw].sum()
        a_tf = res['transport_free'][nm].sum() / res['transport_free'][fw].sum()
        line += f"  {a_ce:.3e} / {a_sl:.3e} / {a_tf:.3e}"
        print(line)
    # headline: deep concrete spectrum, who wins
    deep = "concrete"
    sl = shape_err(res['stochastic_slab'][deep], ref[deep])
    tf = shape_err(res['transport_free'][deep], ref[deep])
    print(f"\nDEEP ({deep}) spectrum shape error vs CE:  slab {sl:.2f}%   transport_free {tf:.2f}%  "
          f"-> {'transport_free WINS' if tf < sl else 'stochastic_slab wins'}")

def plot_geometry():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    model, rc = build_model("continuous-energy")
    mats = {m.name: m for m in model.materials}
    colmap = {"plasma": (255, 235, 150), "tungsten": (90, 90, 110), "steel": (140, 150, 165),
              "lithium": (190, 225, 235), "concrete": (205, 180, 150)}
    colors = {mats[n]: c for n, c in colmap.items()}
    R = LAYOUT[-1][1]
    fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=150)
    model.plot(basis="xy", width=(2*R+40, 2*R+40), pixels=(1400, 1400),
               colors=colors, color_by="material", axes=ax, legend=False)
    ax.set_title("RR-vs-CE fusion shield benchmark (XY slice)\n"
                 "14.1 MeV plasma source -> W armor -> steel -> Li blanket -> concrete", fontsize=11)
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    r_prev = 0.0; handles = []
    for name, r_out, _ in LAYOUT:
        handles.append(mp.Patch(facecolor=np.array(colmap[name])/255.0, edgecolor="k",
                                label=f"{name}  ({r_prev:.0f}-{r_out:.0f} cm)"))
        r_prev = r_out
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax.add_patch(plt.Circle((0, 0), LAYOUT[0][1], fill=False, ls="--", lw=1.2, ec="red"))
    ax.plot(0, 0, marker="*", color="red", ms=14)
    fig.tight_layout(); fig.savefig("rr_benchmark_geometry.png", bbox_inches="tight")
    print("saved rr_benchmark_geometry.png")

if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "ce": run_ce()
    elif stage == "rr": run_rr(sys.argv[2] if len(sys.argv) > 2 else "transport_free")
    elif stage == "compare": compare()
    elif stage == "plot": plot_geometry()
    else: print(__doc__)
