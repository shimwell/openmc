# Transport-free MGXS → random ray: fusion-shield benchmark

A decisive end-to-end test of the `transport_free` MGXS generation method (added to
`Model.convert_to_multigroup(method="transport_free")`): generate a multigroup library
deterministically (no Monte Carlo, no transport solve), feed it to OpenMC's **random ray**
solver, and compare the resulting flux against a **continuous-energy Monte Carlo reference**
on a deep-penetration fusion shield. The same is done with `stochastic_slab` (the existing
MC-based method) so the two can be compared head-to-head against the CE truth.

## Geometry

Spherical shield using the [neutronics-workshop](https://fusion-energy.github.io/neutronics-workshop)
tokamak materials and thicknesses: a 14.1 MeV plasma source in the centre, then a tungsten
armor layer, a steel structural layer, a lithium breeder blanket, and a concrete bioshield.

![benchmark geometry](rr_benchmark_geometry.png)

## Running

```bash
python rr_fusion_bench.py ce                 # continuous-energy reference (Monte Carlo)
python rr_fusion_bench.py rr stochastic_slab # random ray fed by the slab MGXS
python rr_fusion_bench.py rr transport_free  # random ray fed by the transport-free MGXS
python rr_analyze.py                          # volume-averaged comparison vs CE
```

Append a group-structure name (e.g. `CCFE-709`) to any command to change the energy mesh.

## Metrics

CE flux tallies are volume-*integrated* (track length); random ray
`volume_normalized_flux_tallies` are volume-*averaged*. `rr_analyze.py` converts both to a
common volume-averaged region spectrum using analytic shell volumes, then reports:

- **spectrum-shape error** vs CE per region (each spectrum normalized to unit sum —
  normalization-independent),
- **deep-attenuation error**: how accurately the flux suppression deep in the shield
  (φ_region / φ_first-wall) matches CE.

## Result

**Robustness.** `transport_free` produces a library that is **valid in every group**
(positive total XS, no Monte Carlo noise), so it feeds random ray with no fixups.
`stochastic_slab` produced zero / negative total cross sections (unpopulated deep-thermal
groups + a transport-correction artifact in lithium) that random ray rejects; the harness
applies an identical positive-floor fixup to both for a fair comparison. The noise is also
visible directly: in the deep-thermal tail of the near-source regions `stochastic_slab`
swings over tens of decades of Monte Carlo noise while `transport_free` is smooth (see
`rr_spectra_*.png`).

**Accuracy vs CE.** The verdict is group-structure dependent:

- At a **coarse** structure (`VITAMIN-J-42`, 40 groups) `transport_free` is markedly more
  accurate deep in the shield — the deep-attenuation error vs CE is ~41 % versus ~234 % for
  `stochastic_slab`, because the slab's noisy/biased coarse-group cross sections badly
  mis-predict the flux suppression.
- At a **fine fusion** structure (`CCFE-709`, 650 groups) the two methods **converge to
  near-parity** (flux-weighted spectral error within a few % of each other in every region):
  `stochastic_slab` keeps a small edge at the near-source first wall, `transport_free` is
  slightly better at the deepest point. Finer groups make the within-group weighting nearly
  irrelevant, so the methods agree — consistent with the collapse-metric studies on PR #113.

So for the random ray workflow, `transport_free` **matches `stochastic_slab`'s accuracy at
the fine groups used for fusion shielding, beats it at coarse groups, and is strictly more
robust** (deterministic, noise-free, valid in every group).

**Caveat.** The random ray solve here is moderately resolved (a handful of flat source
regions per layer); absolute errors vs CE (~20–40 % flux-weighted at `CCFE-709`) are
dominated by that spatial resolution and are common to both methods. A higher-resolution
random ray solve (more source regions, linear sources, more rays) would sharpen the absolute
accuracy verdict; the *relative* method comparison is unaffected.
