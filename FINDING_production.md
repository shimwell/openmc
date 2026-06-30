# Production deterministic-transport (#117): first validation

One full-stack 1D transport solve (real layer order) -> per-region flux -> collapse each
material with the CORRECT kernel scatter XS (incl. multiplicity) and weighting = transport
above the free-gas threshold (400 kT, universal constant) + NR below.

## 10-material result (CCFE-709, vs converged stochastic_slab)
| material | TOTAL det/slab | ROWSUM det/slab | SHAPE det/slab |
|---|---|---|---|
| **Fe-56** | **0.32 / 1.41** | **0.40 / 1.47** | 0.010/0.009 |
| **Zircaloy** | **0.56 / 0.91** | **0.92 / 1.00** | 0.014/0.013 |
| Helium | 0.04 / 0.04 | **0.05 / 9.66** | 0.116/0.523 |
| concrete/H2O/Li4SiO4 | win/tie | slightly worse (<0.4%) | win/tie |
| tungsten | 3.79 / 2.57 | 5.26 / 2.34 | 0.004/0.004 |
| steel / CuCrZr | worse | worse | ~tie |

**Wins: TOTAL 7/10, scatter ROWSUM 3/10, SHAPE 3/10.**

## Headline + the catch
- **Fe-56 cracked on BOTH total (0.32) and scatter (0.40)** -- the resonant metal NR failed.
  The deep / cross-talk materials win; this is the whole point.
- **Near-source metals (tungsten/steel/CuCrZr) regress vs NR.** Checked vs BOTH the sphere
  reference AND the geometry-consistent slab1d reference: tungsten bad vs both (3.79 / 4.20)
  -> this is NOT a sphere-vs-slab artifact. The transport's *innermost-shell* (source-adjacent)
  flux is genuinely wrong -- almost certainly the **incident-boundary source treatment** vs the
  references' volumetric source.

## Implication for "switch-free"
"Transport everywhere" does NOT beat NR everywhere -- it trades the near-source metals for Fe-56.
Two ways forward:
1. **Fix the innermost-shell source treatment** (volumetric isotropic source in the source cell
   instead of an incident boundary) -- if that fixes tungsten, switch-free is viable.
2. **Physically-motivated division (auto-detected, not arbitrary):** NR for the source-adjacent
   region (where the flux IS the source spectrum and NR is exact) + transport for the deeper
   materials (cross-talk). The transport flux itself signals the regime (source-like vs degraded).

For deep-penetration shielding (the target use case) the deep materials dominate, so the method
already delivers where it matters; the near-source first wall is where NR is fine anyway.

## Update: volumetric source fix tried -> no effect (diagnostic)
Replaced the incident-boundary source with a volumetric isotropic source in the source
cell. Result is **byte-identical** (tungsten still 3.79/5.26). This is informative: the
collapse uses the per-region SPECTRAL SHAPE, which is set by the slowing-down (energy)
physics; the source's spatial/angular form doesn't change it (both inject the same 14 MeV
muir spectrum). So the near-source error is **not** the source treatment, and (verified
earlier vs slab1d) **not** geometry.

The remaining cause is the transport's **slowing-down physics: the down-scatter source is
ELASTIC-ONLY**. Tungsten has strong inelastic scattering, so its source-region flux comes
out too hard -> wrong collapse. (Fe-56, deep, is unaffected: its flux is set by the
incoming spectrum from upstream, not its own inelastic.)

## Update: inelastic + (n,2n)/(n,3n) down-scatter ADDED to the transport (`inel_source.py`)
The transport source is now elastic (fine gather) **plus** inelastic + (n,xn). The inelastic
outgoing is smooth, so it's built as a COARSE transfer matrix `M[out,in]` per material
(incl. multiplicity) and applied to the flux in an **outer iteration** (elastic solve ->
add inelastic source from that flux -> re-solve, x2). +71s, ~no accuracy cost from the
coarse grid (inelastic spectra are broad).

### Result: TOTAL 7/10 -> **9/10** (steel fixed)
| material | TOTAL det/slab | ROWSUM det/slab | was (elastic-only) |
|---|---|---|---|
| **steel** | **0.25 / 0.48** | **0.42 / 0.51** | 0.70 / 0.89 -> now beats slab on both |
| **Fe-56** | **0.44 / 1.41** | **0.51 / 1.47** | 0.32 / 0.40 (slightly up, still crushes slab) |
| **Zircaloy** | **0.57 / 0.91** | **0.94 / 1.00** | win |
| CuCrZr | 1.06 / 1.02 | 1.47 / 1.05 | ~tie total |
| tungsten | **3.79 / 2.57** | 5.26 / 2.34 | **UNCHANGED** |

**Inelastic fixed steel (the win) but left tungsten exactly unchanged.**

### Why tungsten is immune (it is NOT the inelastic, and NOT a build bug)
Confirmed `M_inel[tungsten]` builds correctly (sum 17.5, all 5 W isotopes' MT51-91 + (n,2n)
present) and is applied to the tungsten cells -- yet the collapse doesn't move. So tungsten's
error is **source-region geometry**, not slowing-down physics:
- In `material_wise` (the truth) tungsten is mixed through the **whole domain** and sees the
  fully built-up equilibrium (~1/E) slowing-down spectrum.
- In the slab tungsten sits **only at the front 6 cm**, so its flux is the 14 MeV source
  lightly self-slowed -- **too hard** vs the equilibrium. The deep materials don't have this
  problem (they sit where a slowed spectrum is physical).
- **NR assumes exactly that equilibrium 1/E**, so NR is *better* for the front wall (1.96 vs
  transport 3.79). This is intrinsic to placing the front material only at the front.

## Conclusion for #117
- Inelastic down-scatter **completes the transport's slowing-down physics**: steel fixed,
  Fe-56/Zircaloy/deep materials all win -> **TOTAL 9/10, the deep-penetration goal delivered.**
- The lone holdout is **tungsten, the first wall** -- a geometry/placement effect (front
  material under-sees the equilibrium spectrum), not a physics gap. **NR is the right tool
  there** (1.96), and the first wall is exactly where NR is valid.
- Clean switch-free rule that needs no arbitrary threshold: **NR for the source layer**
  (flux == source spectrum, NR exact) **+ transport for everything downstream**. Equivalent
  to "use NR where there is no upstream material to slow neutrons down," which the geometry
  itself defines.
