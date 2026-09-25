# Finding: a deterministic stochastic_slab reduces to NR

Goal (Option A): reproduce OpenMC's `stochastic_slab` representative weighting
deterministically (reflective box of mixed materials + volumetric source).

## Result: it collapses to narrow-resonance (NR), not to slab
Weighting each material X by phi_X(E) = q_mix(E)/Sigma_t,X(E) — the material-mix
slowing-down source x X's own self-shielding — gives, vs material_wise (CCFE-709):

| material | dslab | slab | NR |
|---|---|---|---|
| tungsten | 1.77 | 2.56 | 1.96 |
| Fe-56 | 3.28 | 1.39 | 3.26 |
| steel | 0.60 | 0.48 | 0.57 |
| Li4SiO4 | 0.07 | 0.42 | 0.06 |

dslab ≈ NR everywhere.

## Why
1. The mix slowing-down source q_mix(E) is ≈ 1/E (asymptotic slowing-down density is
   ~constant per lethargy for any moderating mix), so phi_X = q_mix/Sigma_t,X = NR.
2. Slab's actual edge over NR is NOT the spectrum — it is the sigma0-DILUTION of its
   finite (1 cm) mixed cells (equivalence-theory self-shielding: neutrons inflow from
   neighbouring non-resonant cells reduce resonance self-shielding). That is tied to
   slab's arbitrary cell size — partly a numerical artifact.

## Conclusion
You cannot beat NR on deep resonant metals in an order-free / geometry-free way — the
improvement (cross-talk + finite-region escape self-shielding) is inherently
geometry-dependent. The physically-correct version is the real-geometry deterministic
transport (PR #114), which beats slab (Fe-56 0.32%) where this approach cannot.

Recommendation: do not pursue this branch as a separate method; it is the NR
transport-free collapse (PR #113). Focus on #114 (real-geometry transport).
