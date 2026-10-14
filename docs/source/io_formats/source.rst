.. _io_source:

==================
Source File Format
==================

Normally, source data is stored in a state point file. However, it is possible
to request that the source be written separately, in which case the format used
is that documented here.

When surface source writing is triggered, a source file named
``surface_source.h5`` is written with only the sources on specified surfaces,
following the same format.

**/**

:Attributes: - **filetype** (*char[]*) -- String indicating the type of file.
             - **version** (*int[2]*) -- Major and minor version of the source
               file format.

:Datasets:

           - **source_bank** (Compound type) -- Source bank information for each
             particle. The compound type has fields ``r``, ``u``, ``E``,
             ``time``, ``wgt``, ``delayed_group``, ``surf_id`` and ``particle``,
             which represent the position, direction, energy, time, weight,
             delayed group, surface ID, and particle type (PDG number),
             respectively.

Optional FileSource replay datasets
===================================

The ordinary ``source_bank`` compound layout is unchanged. Source files without
the extensions below remain compatible with :class:`openmc.FileSource`: sites
are sampled uniformly and retain their stored ``wgt``. Parent tags and sampling
weights are independent optional extensions; neither changes the compound type.

**/**

:Datasets:

           - **parent_names** (*string[]*) -- One-dimensional array of decay
             parent nuclide names. Fixed-width and variable-length HDF5 strings
             are supported. Every name must occur in the loaded reaction/decay
             chain, for example supplied through ``OPENMC_CHAIN_FILE``.
           - **parent_nuclide** (*int[]*) -- One zero-based index into
             ``parent_names`` per source site, in ``source_bank`` order. Both
             parent datasets must be present if either is present. Names are
             resolved to the loaded chain's indices at replay, so model-local
             chain ordering need not be preserved. These tags enable
             :class:`openmc.ParentNuclideFilter` and are inherited by split
             particles and secondary tracks, including bremsstrahlung photons;
             they are not inventory data.
           - **sampling_weights** (*double[]*) -- One finite, nonnegative
             sampling weight per source site, with a positive total. When
             present, sites are sampled in proportion to these weights and
             each sampled site is assigned ``wgt=1``, regardless of its stored
             ``wgt``. Do not multiply the stored birth weight into the sampled
             particle again. The weights define a probability distribution,
             not automatic absolute-rate normalization. Supply the physical
             rate separately through ``FileSource.strength`` or normalize a
             unit-strength response once, not both.

An empty bank cannot be sampled by ``FileSource``. An empty D1S capture shard,
however, still contributes its ``source_histories`` to the global history count
when shards are combined.

.. _io_source_d1s:

D1S capture files
=================

With :ref:`io_settings_d1s_source_write`, the executable writes one
``d1s_source.<rank>.h5`` per MPI rank (``d1s_source.0.h5`` in serial), under
the configured output directory. Even a rank with zero births writes a file.
Each contains the ordinary ``source_bank`` with photon birth positions (cm),
unit directions, energies (eV), and sampled production weights. Capture sets
source times to zero; these are not cumulative irradiation/cooling times.
Parent tags are stored separately, not as an extra compound member. The capture
schema has the following additional attributes and datasets:

**/**

:Attributes: - **filetype** (*char[]*) -- ``source``.
             - **version** (*int[2]*) -- The ordinary source/statepoint format
               version, separate from the D1S extension version.
             - **d1s_version** (*int*) -- D1S extension version, currently 1.
               Version 1 capture files require ``source_histories``.

:Datasets:

           - **parent_names** (*string[]*) -- Names from the capture chain.
           - **parent_nuclide** (*int[]*) -- Zero-based parent-name indices,
             one per birth.
           - **cell_id** (*int[]*) -- Emitting leaf cell's runtime OpenMC ID.
           - **cell_instance** (*int[]*) -- Zero-based occurrence of that leaf
             cell, including repeated universe/lattice placements. For CSG,
             numbering follows the backend's canonical XML traversal, with
             cells ordered by ascending ID within each universe, not Python
             cell-insertion order.
           - **universe_id** (*int[]*) -- Emitting leaf universe's OpenMC ID.
           - **dagmc_volume_id** (*int[]*) -- Raw H5M volume ``GLOBAL_ID`` for
             a DAGMC birth, or zero for a CSG birth. Interpret raw IDs within
             ``universe_id``, not as runtime cell IDs assigned by
             ``auto_geom_ids``.
           - **history_id** (*int64[]*) -- Emitting transport track's particle
             ID (``p.id()``). This is not guaranteed to identify the original
             external-source history for shared-bank secondary tracks.
           - **batch** (*int[]*) -- Originating transport batch number.
           - **source_histories** (*int64*) -- Rank-local count of all sampled
             original external neutron histories over the capture calculation,
             including zero-weight histories rejected with the ``kill``
             strategy. Secondary and split tracks are excluded. This required
             scalar is a signed 64-bit integer, present even for an empty bank,
             and contributes to the global production normalization
             denominator.
           - **source_weight** (*double*) -- Rank-local sum of actual original
             external neutron source weights over the capture calculation,
             before initial weight-window splitting/roulette and excluding
             secondary and split tracks. Kill-rejected histories contribute
             zero. This scalar is informational; it is neither the photon
             weight sum nor the count of captured births.

All per-birth arrays are one-dimensional and aligned with ``source_bank``.
The ``history_id`` provenance limitation does not affect ``source_histories``:
that scalar independently counts original external neutron histories and
remains the correct production normalization count with shared-bank
secondaries.
Capture requires neutron-only external sources; a sampled non-neutron site is
an error even if its weight is zero. For global production normalization, sum
``source_histories`` over **all** rank files, including those with empty banks.
Divide each birth weight by this global history count, then apply the
irradiation/decay factors and physical initial neutron source rate. This
matches fixed-source normalization per initial source history. Nonunit source
sampling-importance weights remain in the birth weights; do not divide by their
sum. Counting kill-rejected histories preserves the source rejection
probability. The capture files contain no schedule-specific source rate and do
not by themselves represent photons/s. Storage and capture memory grow with the
number of sampled births; the bank is not streamed in bounded-size chunks.

D1S composed replay files
=========================

:meth:`openmc.deplete.d1s.Simulation.evaluate` writes a composed ``photons.h5``
for each nonzero cooling observation. It retains ``filetype``, ``version``, and
``d1s_version=1`` and includes ``parent_names``, ``parent_nuclide``, and
``sampling_weights`` as described above. Stored site weights are one and
originating surface IDs are cleared. Each sampling weight is a physical
emission contribution in photons/s after time weighting and component removal;
their sum is the observation's total photon rate. The workflow also records:

**/**

:Attributes:

           - **d1s_target_selectors** (*string*) -- JSON-encoded destination
             selectors used by the isolated DAGMC geometry preflight, or an
             empty list for CSG-only target geometry.

:Datasets:

           - **photon_source_rate** (*double*) -- Sum of ``sampling_weights``,
             in photons/s. ``FileSource`` does not read this scalar to set its
             strength automatically.
           - **component_names** (*string[]*) -- Target component identities.
           - **target_component** (*int64[]*) -- Per-site zero-based index
             into ``component_names``, or -1 in no-map mode.
           - **kernel_index** (*int64[]*) -- Per-site index of the contributing
             production kernel for this observation.

The capture arrays ``cell_id``, ``cell_instance``, ``universe_id``,
``dagmc_volume_id``, ``history_id``, and ``batch`` are retained as *origin*
metadata, not rewritten as destination IDs. These extra provenance datasets do
not change the ``FileSource`` sampling algorithm. Retained ``history_id``
values have the same transport-track provenance limitation as in capture files.
Photon replay and the geometry locator clear any retained ``surf_source_read``
setting so it cannot append an extra external source.

Zero-rate observations write no replay bank or statepoint. Sizing zero tallies
still requires an isolated geometry-only Python C API subprocess for a target
containing DAGMC with a :class:`openmc.DistribcellFilter`, or for any target
with an unstructured-mesh tally. The shared library must support DAGMC or the
selected unstructured-mesh backend, respectively, even when no transport runs.
This subprocess initializes neither sources nor transport and requires no
nuclear cross-section or decay-chain data. Prepared production is an
in-process object; these files are not a general restart format for
:class:`~openmc.deplete.d1s.Production`. See
:ref:`usersguide_d1s_multi_model` for component transfer and physical tally
normalization.
