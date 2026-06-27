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
