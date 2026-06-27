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
