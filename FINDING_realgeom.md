# Real-geometry deterministic transport: status

Goal: solve the deterministic transport in the ACTUAL reference geometry (concentric
spheres, central point source) so the near-source shells get the correct (harder) flux
that the 1D-slab prototype (#114) over-softens.

## Built + validated: 1D spherical Sn (sphere_sn.py)
Weighted-diamond curvilinear Sn with the angular-redistribution recursion + central
point source. **Validated against the analytic point-source-in-uniform-absorber flux
phi(r)=S exp(-Sigma_t r)/(4 pi r^2): 0.9% mean, 1.6% max** -- the curvilinear physics
(streaming + angular redistribution + central source) is correct.

## Open issue: diamond-difference negativity on the resonant problem
On the real material stack (resonances -> optically-thick cells + the alpha angular-
redistribution term), the diamond scheme produces negative/oscillating fluxes:
  steel TOTAL 7.37 (NR 0.57!), broken scatter SHAPE 0.76.
Naive clamping (set psi>=0) breaks conservation and doubles the smooth-case flux (ratio
2.0) -- it is NOT a valid fixup. A positivity-preserving CURVILINEAR scheme is required
(step characteristic, or weighted-diamond with a proper negative-flux fixup that also
treats the angular edges). This is the next engineering step.

## Where it stands
- Deep/cross-talk materials are already won by the 1D-SLAB transport (#114): Fe-56 0.32,
  H2O/Li4SiO4/concrete/He all beat slab, deterministically.
- The spherical solver is needed only to also fix the NEAR-SOURCE shells (tungsten,
  steel) -- which is exactly where NR is already excellent (<=2%).
- Pragmatic alternatives to the full positivity-preserving spherical Sn:
  (a) hybrid: NR for near-source shells (hard, source-dominated) + slab transport for the
      deep shells (cross-talk) -- the local flux itself signals the regime;
  (b) ship #113 (NR) as the robust geometry-independent library and #114 (slab transport)
      as the optional deep-material enhancement.

## Update: positivity-preserving step-characteristic scheme added
The curvature coefficients alpha_{m+1/2} are provably >= 0 (a tent: 0 -> peak -> 0),
which makes STEP-CHARACTERISTIC differencing (upwind in space AND angle)
UNCONDITIONALLY POSITIVE for spherical Sn. Implemented as scheme='step' (default) in
sphere_sn.py. Validation (point source in absorber): step 1.2% mean (diamond 0.9%),
min flux > 0 (vs diamond which can go negative).

Re-run on the resonant stack (CCFE-709, total %err vs material_wise):
| material | NR | diamond(broken) | STEP | slab |
|---|---|---|---|---|
| tungsten | 1.96 | 5.51 | **2.13** | 2.56 |
| steel | 0.57 | 7.37 | 2.44 | 0.48 |
| Fe-56 | 3.26 | (neg) | **1.08** | 1.39 |
| CuCrZr | 0.85 | (neg) | 1.20 | 1.02 |

Step fixes the negativity (no more catastrophic steel 7.37) and tungsten + Fe-56 now
BEAT slab. BUT step is 1st-order diffusive, so steel/CuCrZr are worse than NR/slab.
The diamond scheme is 2nd-order accurate but unstable on resonances. Neither is a clean
win: the next step for a uniform spherical solver is **diamond with a proper negative-
flux fixup** (set-to-zero + re-solve the cell, conserving), which keeps 2nd-order
accuracy AND positivity.

## Practical comparison of paths (CCFE-709)
- The 1D-SLAB transport (#114) + thermal-NR fallback is the most accurate for the DEEP
  materials: Fe-56 total 0.60 / scatter 0.69 (both beat slab) -- but slab geometry
  over-softens the NEAR-SOURCE shells.
- This spherical-STEP solver fixes near-source tungsten (2.13 < slab 2.56) but its
  diffusivity hurts steel/CuCrZr.
- A diamond+fixup spherical solver should get the best of both (accurate + positive +
  correct geometry) -- the recommended future direction for a single uniform method.

## Update 2: diamond + negative-flux fixup added (scheme='fixup', now default)
Set-to-zero & re-solve: where a diamond cell's outgoing spatial/angular edge would go
negative, clamp it to 0 and re-solve the cell (the curvature coeffs alpha>=0 guarantee
termination positive). Validation (point source in absorber): **fixup 0.74% mean (best;
diamond 0.89, step 1.17), positive flux.**

BUT on the RESONANT stack (CCFE-709, total %err vs material_wise):
| material | step | fixup | slab-transport(#114) | slab(MC) |
|---|---|---|---|---|
| tungsten | 2.13 | 2.47 | (5.51 slab-geom) | 2.56 |
| steel    | 2.44 | 2.53 | 0.57(NR) | 0.48 |
| Fe-56    | 1.08 | 2.06 | **0.60** | 1.39 |
| CuCrZr   | 1.20 | 1.63 | 0.85(NR) | 1.02 |

**fixup is WORSE than step on resonances** -- at the many resonance negativities the
set-to-zero clamp accumulates more error than step's consistent upwinding. So for the
resonant deep-penetration problem, neither spherical scheme beats the simpler, robust
1D-SLAB transport (#114, diamond + edge clamp): Fe-56 0.60 (slab-transport) vs 1.08
(spherical step). The spherical geometry's only clear gain is the near-source shell
(tungsten 2.13 vs slab-method 2.56), where NR is already excellent anyway.

## Bottom line / recommendation
A correct-geometry spherical Sn does NOT outperform the robust slab transport on the
deep resonant materials -- the spherical central-source + resonance negativity make it
numerically harder, and the schemes that are positive (step/fixup) are too diffusive or
clamp-lossy. The practical best method remains:
  * #113 (NR) as the geometry-free default, +
  * #114 (1D-slab transport + thermal-NR fallback) for the deep cross-talk materials
    (Fe-56 total 0.60 / scatter 0.69, both beat slab),
  * NR for the near-source shells (already <=2%).
A uniform high-accuracy positive spherical solver would need more advanced numerics
(characteristic/CN, or much finer mesh) -- diminishing returns vs the hybrid above.
