.. usersguide_decay_sources:

=============
Decay Sources
=============

OpenMC can simulate radiation emitted from the decay of activated materials.
For fusion energy systems, this is commonly done using either the `rigorous
2-step <https://doi.org/10.1016/S0920-3796(02)00144-8>`_ (R2S) method or the
`direct 1-step <https://doi.org/10.1016/S0920-3796(01)00188-0>`_ (D1S) method.
In the R2S method, a neutron transport calculation is used to determine the
neutron flux and reaction rates over a cell- or mesh-based spatial
discretization of the model. Then, the neutron flux in each discrete region is
used to predict the activated material composition using a depletion solver.
Finally, a photon transport calculation with a source based on the activity and
energy spectrum of the activated materials is used to determine a desired
physical response (e.g., a dose rate) at one or more locations of interest.
OpenMC includes automation for both methods. R2S uses
:ref:`depletion <usersguide_depletion>` to calculate an inventory; D1S weights
reaction-produced decay photons without calculating an inventory or changing
material compositions.

Rigorous 2-Step (R2S) Calculations
==================================

OpenMC includes an :class:`openmc.deplete.R2SManager` class that fully automates
cell- and mesh-based R2S calculations. Before we describe this class, it is
useful to understand the basic mechanics of how an R2S calculation works.
Generally, it involves the following steps:

1. The :meth:`openmc.deplete.get_microxs_and_flux` function is called to run a
   neutron transport calculation that determines fluxes and microscopic cross
   sections in each activation region.
2. The :class:`openmc.deplete.IndependentOperator` and
   :class:`openmc.deplete.PredictorIntegrator` classes are used to carry out a
   depletion (activation) calculation in order to determine predicted material
   compositions based on a set of timesteps and source rates.
3. The activated material composition is determined using the
   :class:`openmc.deplete.Results` class. Indexing an instance of this class
   with the timestep index returns a :class:`~openmc.deplete.StepResult` object,
   which itself has a :meth:`~openmc.deplete.StepResult.get_material` method
   returning an activated material.
4. The :meth:`openmc.Material.get_decay_photon_energy` method is used to obtain
   the energy spectrum of the decay photon source. The integral of the spectrum
   also indicates the intensity of the source in units of [Bq].
5. A new photon source is defined using one of OpenMC's source classes with the
   energy distribution set equal to the object returned by the
   :meth:`openmc.Material.get_decay_photon_energy` method. The source is then
   assigned to a photon :class:`~openmc.Model`.
6. A photon transport calculation is run with ``model.run()``.

Altogether, the workflow looks as follows::

    # Run neutron transport calculation
    fluxes, micros = openmc.deplete.get_microxs_and_flux(model, domains)

    # Run activation calculation
    op = openmc.deplete.IndependentOperator(mats, fluxes, micros)
    timesteps = ...
    source_rates = ...
    integrator = openmc.deplete.Integrator(op, timesteps, source_rates)
    integrator.integrate()

    # Get decay photon source at last timestep
    results = openmc.deplete.Results("depletion_results.h5")
    step = results[-1]
    activated_mat = step.get_material('1')
    photon_energy = activated_mat.get_decay_photon_energy()
    photon_source = openmc.IndependentSource(
        space=...,
        energy=photon_energy,
        particle='photon',
        strength=photon_energy.integral()
    )

    # Run photon transport calculation
    model.settings.source = photon_source
    model.run()

Note that by default, the :meth:`~openmc.Material.get_decay_photon_energy`
method will eliminate spectral lines with very low intensity, but this behavior
can be configured with the ``clip_tolerance`` argument.

Cell-based R2S
--------------

In practice, users do not need to manually go through each of the steps in an R2S
calculation described above. The :class:`~openmc.deplete.R2SManager` fully
automates the execution of neutron transport, depletion, decay source
generation, and photon transport. For a cell-based R2S calculation, once you
have a :class:`~openmc.Model` that has been defined, simply create an instance
of :class:`~openmc.deplete.R2SManager` by passing the model and a list of cells
to activate::

    r2s = openmc.deplete.R2SManager(model, [cell1, cell2, cell3])

Note that the ``volume`` attribute must be set for any cell that is to be
activated. The :class:`~openmc.deplete.R2SManager` class allows you to
optionally specify a separate photon model; if not given as an argument, it will
create a shallow copy of the original neutron model (available as the
``neutron_model`` attribute) and store it in the ``photon_model`` attribute. We
can use this to define tallies specific to the photon model::

    dose_tally = openmc.Tally()
    ...
    r2s.photon_model.tallies = [dose_tally]

Next, define the timesteps and source rates for the activation calculation::

    timesteps = [(3.0, 'd'), (5.0, 'h')]
    source_rates = [1e12, 0.0]

In this case, the model is irradiated for 3 days with a source rate of
:math:`10^{12}` neutron/sec and then the source is turned off and the activated
materials are allowed to decay for 5 hours. These parameters should be passed to
the :meth:`~openmc.deplete.R2SManager.run` method to execute the full R2S
calculation. Before we can do that though, for a cell-based calculation, the one
other piece of information that is needed is bounding boxes of the activated
cells::

    bounding_boxes = {
        cell1.id: cell1.bounding_box,
        cell2.id: cell2.bounding_box,
        cell3.id: cell3.bounding_box
    }

Note that calling the ``bounding_box`` attribute may not work for all
constructive solid geometry regions (for example, a cell that uses a
non-axis-aligned plane). In these cases, the bounding box will need to be
specified manually. Once you have a set of bounding boxes, the R2S calculation
can be run::

    r2s.run(timesteps, source_rates, bounding_boxes=bounding_boxes)

Neutron tally results from the transport step are available in the
``r2s.results['neutron_tallies']`` list, in the same order as the tallies on
``r2s.neutron_model``. Photon tally results remain available through
``r2s.results['photon_tallies']``.

If not specified otherwise, a photon transport calculation is run at each time
in the depletion schedule for which a decay photon source exists. Times without
a decay photon source, such as the initial state of a model containing only
stable nuclides, are omitted. To specify particular times at which photon
transport calculations should be run, pass the ``photon_time_indices`` argument.
For example, if we wanted to run a photon transport calculation only on the last
time (after the 5 hour decay), we would run::

    r2s.run(timesteps, source_rates, bounding_boxes=bounding_boxes,
            photon_time_indices=[2])

To attribute photon tally results to their parent radionuclides, set
``by_parent_nuclide=True``. This automatically adds a
:class:`openmc.ParentNuclideFilter` to every photon tally that does not already
have one. The filter bins are the union of radionuclides contributing to the
prepared decay photon sources. The resulting bins can be used directly when
inspecting the tally results::

    r2s.run(timesteps, source_rates, bounding_boxes=bounding_boxes,
            photon_time_indices=[2], by_parent_nuclide=True)

    photon_tally = r2s.results['photon_tallies'][2][0]
    tally_by_parent = photon_tally.get_pandas_dataframe()

After an R2S calculation has been run, the :class:`~openmc.deplete.R2SManager`
instance will have a ``results`` dictionary that allows you to directly access
results from each of the steps. It will also write out all the output files into
a directory that is named "r2s_<timestamp>/". The ``output_dir`` argument to the
:meth:`~openmc.deplete.R2SManager.run` method enables you to override the
default output directory name if desired.

The :meth:`~openmc.deplete.R2SManager.run` method actually runs four
lower-level methods under the hood::

    r2s.step1_neutron_transport(...)
    r2s.step2_activation(...)
    r2s.step3_photon_source(...)
    r2s.step4_photon_transport(...)

For users looking for more control over the calculation, these lower-level
methods can be used in lieu of the :meth:`openmc.deplete.R2SManager.run` method.

Mesh-based R2S
--------------

Executing a mesh-based R2S calculation looks nearly identical to the cell-based
R2S workflow described above. The only difference is that instead of passing a
list of cells to the ``domains`` argument of
:class:`~openmc.deplete.R2SManager`, you need to define a mesh object and pass
that instead. This might look like the following::

    # Define a regular Cartesian mesh
    mesh = openmc.RegularMesh()
    mesh.lower_left = (-50., -50., 0.)
    mesh.upper_right = (50., 50., 75.)
    mesh.dimension = (10, 10, 5)

    r2s = openmc.deplete.R2SManager(model, mesh)

Executing the R2S calculation is then performed by adding photon tallies and
calling the :meth:`~openmc.deplete.R2SManager.run` method with the appropriate
timesteps and source rates. Note that in this case we do not need to define cell
volumes or bounding boxes as is required for a cell-based R2S calculation.
Instead, during the neutron transport step, OpenMC will run a raytracing
calculation to determine material volume fractions within each mesh element
using the :meth:`openmc.MeshBase.material_volumes` method. Arguments to this
method can be customized via the ``mat_vol_kwargs`` argument to the
:meth:`~openmc.deplete.R2SManager.run` method. Most often, this would involve
customizing the number of rays traced to obtain better estimates of volumes. As
an example, if we wanted to run the raytracing calculation with 10 million rays,
we would run::

    r2s.run(timesteps, source_rates, mat_vol_kwargs={'n_samples': 10_000_000})

It is also possible to use multiple meshes by passing a list of meshes instead
of a single mesh. This can be useful, for example, when different regions of the
model require different mesh resolutions. The meshes are assumed to be
**non-overlapping**; each element--material combination across all meshes is
treated as an independent activation region, and all meshes are handled in a
single neutron transport solve. For example::

    # Fine mesh near the activation target
    mesh_fine = openmc.RegularMesh()
    mesh_fine.dimension = (10, 10, 10)
    ...

    # Coarse mesh for the surrounding region
    mesh_coarse = openmc.RegularMesh()
    mesh_coarse.dimension = (5, 5, 5)
    ...

    r2s = openmc.deplete.R2SManager(model, [mesh_fine, mesh_coarse])

.. _usersguide_d1s:

Direct 1-Step (D1S) Calculations
================================

OpenMC also includes built-in capability for performing shutdown dose rate
calculations using the `direct 1-step
<https://doi.org/10.1016/S0920-3796(01)00188-0>`_ (D1S) method. In this method,
a single coupled neutron--photon transport calculation is used where the prompt
photon production is replaced with photons produced from the decay of
radionuclides in an activated material. To obtain properly scaled results, it is
also necessary to apply time correction factors. A normal neutron transport
calculation can be extended to a D1S calculation with a few helper functions.
First, import the ``d1s`` submodule, which is part of :mod:`openmc.deplete`::

    from openmc.deplete import d1s

You need to instruct OpenMC to use decay photon data instead of prompt
photon data. This is done with an attribute on the :class:`~openmc.Settings`
class::

    model = openmc.Model()
    ...
    model.settings.photon_transport = True
    model.settings.use_decay_photons = True

To prepare any tallies for use of the D1S method, you should call the
:func:`~openmc.deplete.d1s.prepare_tallies` function, which adds a
:class:`openmc.ParentNuclideFilter` (used later for assigning time correction
factors) to any applicable tally and returns a list of possible radionuclides
based on the :ref:`chain file <usersguide_data>`. Once the tallies are prepared,
the model can be simulated::

    nuclides = d1s.prepare_tallies(model)
    output_path = model.run()

Finally, the time correction factors need to be computed and applied to the
relevant tallies. This can be done with the aid of the
:func:`~openmc.deplete.d1s.time_correction_factors` and
:func:`~openmc.deplete.d1s.apply_time_correction` functions::

    # Compute time correction factors based on irradiation schedule
    factors = d1s.time_correction_factors(nuclides, timesteps, source_rates)

    # Get tally from statepoint
    with openmc.StatePoint(output_path) as sp:
        dose_tally = sp.get_tally(name='dose tally')

    # Apply time correction factors
    tally = d1s.apply_time_correction(dose_tally, factors, time_index)

The helper workflow above uses one neutron/photon transport configuration.
The capture/replay interface below supports different neutron production states
and a separate photon transport model. Do not apply
:func:`~openmc.deplete.d1s.apply_time_correction` again to capture/replay results.

.. _usersguide_d1s_multi_model:

Staged, multi-model D1S
-----------------------

:class:`openmc.deplete.d1s.Simulation` captures sampled decay-photon births in
neutron calculations, then replays time-weighted births in photon calculations.
It is **not depletion**: compositions and reaction targets remain fixed within
each supplied model. It does not solve a general transmutation/decay-chain
inventory, account for burnup or target depletion, or infer activation before
the schedule. The initial production history is zero. The approximation is
appropriate only when the supplied reaction/decay data and fixed-composition
production kernels adequately represent the activation problem.

Production is piecewise constant over each
:class:`~openmc.deplete.d1s.Irradiation` interval. A configuration specifies
geometry, materials, and neutron source shape, including energy and angular
distributions. Reuse the same
:class:`~openmc.deplete.d1s.Configuration` object when only the source rate or
duration changes. A different neutron spectrum, source distribution, shielding,
or material state requires a different configuration and production
calculation.
``Irradiation.source_rate`` is the total initial neutron source rate in
neutrons/s, not a local flux in neutrons/cm\ :sup:`2`/s. Relative strengths of
multiple neutron sources belong to the model; the schedule supplies their total
physical rate. Production accepts only neutron external sources; mixed
neutron/photon sources are rejected.

For a parent with decay constant :math:`\lambda_p`, a completed irradiation
interval :math:`k`, ending at :math:`b_k`, contributes the factor

.. math::

    F_{k,p}(t) = R_k \left(1-e^{-\lambda_p\Delta t_k}\right)
                e^{-\lambda_p(t-b_k)}.

Here :math:`R_k` is the neutron source rate and :math:`t` is cumulative
schedule time, not time since the most recent configuration change.
Contributions from all preceding completed irradiation intervals are summed,
including repeated uses of a configuration. Cooling produces no new
activation. A
:class:`~openmc.deplete.d1s.Cooling` sample time is a *local* offset in
``[0, duration]``; zero observes the state immediately after switching to the
cooling configuration. The default empty ``sample_times`` requests no
observations. Durations are positive, and times can be seconds or ``(value,
unit)`` pairs with units ``'s'``, ``'min'``, ``'h'``, ``'d'``, or ``'a'``
(Julian years).

CSG components and a staged example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stable component names identify physical objects across models, independently
of model-local cell or material IDs. :class:`~openmc.deplete.d1s.Cells` selects
CSG leaf material or distributed-material cells, not containers, void cells, or
DAGMC cells. Cell objects must belong to the configuration's model; positive
cell IDs are also accepted. Every captured birth must belong to exactly one
non-``None`` component selector. Include all emitting regions, not just regions
whose response is of interest. Passive shielding need not have a component key.

This example uses two independently built neutron models with different source
energies, followed by a third model with the target translated and rotated for
photon transport. It requires neutron/photon cross sections and a configured
``openmc.config['chain_file']`` containing suitable reaction/decay data::

    import numpy as np
    import openmc
    from openmc.deplete import d1s

    def target_model(x=0.0):
        nickel = openmc.Material()
        nickel.add_element('Ni', 1.0)
        nickel.set_density('g/cm3', 8.9)
        boundary = openmc.Sphere(x0=x, r=5.0, boundary_type='vacuum')
        target = openmc.Cell(fill=nickel, region=-boundary)
        model = openmc.Model(geometry=openmc.Geometry([target]))
        model.settings.run_mode = 'fixed source'
        model.settings.energy_mode = 'continuous-energy'
        model.settings.batches = 10
        model.settings.particles = 10_000
        return model, target

    neutron_high, target_high = target_model()
    neutron_low, target_low = target_model()
    photon_model, target_photon = target_model(x=25.0)
    neutron_high.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        energy=openmc.stats.Discrete([14.1e6], [1.0]))
    neutron_low.settings.source = openmc.IndependentSource(
        space=openmc.stats.Point(),
        energy=openmc.stats.Discrete([2.45e6], [1.0]))

    flux = openmc.Tally(name='photon flux')
    flux.filters = [openmc.ParticleFilter('photon'),
                    openmc.CellFilter(target_photon)]
    flux.scores = ['flux']
    photon_model.tallies = [flux]

    high = d1s.Configuration(
        neutron_high, components={'target': d1s.Cells([target_high])})
    low = d1s.Configuration(
        neutron_low, components={'target': d1s.Cells([target_low])})
    placement = np.array([[0., -1., 0., 25.],
                          [1.,  0., 0.,  0.],
                          [0.,  0., 1.,  0.],
                          [0.,  0., 0.,  1.]])
    shutdown = d1s.Configuration(
        photon_model, components={'target': d1s.Cells([target_photon])},
        transforms={'target': placement})

    simulation = d1s.Simulation([
        d1s.Irradiation(high, (1., 'd'), 1.e12, name='high'),
        d1s.Irradiation(low, (2., 'h'), 1.e10, name='low'),
        d1s.Cooling(shutdown, (1., 'd'),
                    sample_times=[0., (1., 'h'), (1., 'd')], name='shutdown'),
    ])
    simulation.validate()  # No transport; does not verify backend capabilities
    production = simulation.prepare(
        output_dir='d1s-production', run_kwargs={'threads': 2})
    results = simulation.evaluate(
        production, output_dir='d1s-evaluation', run_kwargs={'threads': 2})
    observation = results.observations['shutdown', 3600.0]
    photon_rate = observation.source_rate  # photons/s
    tally = results.get_tally('photon flux', 'shutdown', (1., 'h'))

The returned tally is already normalized to the physical photon rate, as
described below.
Alternatively, ``simulation.run(output_dir='d1s', run_kwargs={'threads': 2})``
performs both stages and returns :class:`~openmc.deplete.d1s.Results`.
All output directories must be new or empty; each production kernel and
observation has a dedicated subdirectory.

:class:`~openmc.deplete.d1s.Production` retains captured arrays in memory and
can be evaluated by another simulation with different rates, durations, cooling
models, or sample times. Positive-rate irradiation steps must reference the
*same configuration objects* used in preparation and use the same decay-data
file. Do not mutate configurations, selectors, or their models after preparing
production. This is in-process reuse, not an on-disk restart API. Capturing and
composing banks uses memory of order :math:`O(N)` in the number of captured
births; it is not a bounded-memory streaming calculation.

Regular-mesh animations
~~~~~~~~~~~~~~~~~~~~~~~

Two runnable examples in ``examples/d1s_moving`` separate neutron-source
movement from movement of an activated component:

* ``moving_neutron_source.py`` keeps geometry fixed and irradiates it from
    three neutron-source positions, then observes cooling in the same geometry.
* ``moving_activated_volume.py`` keeps the neutron source and surrounding
    geometry fixed while a single nickel volume changes placement during
    irradiation and cooling. The same component identity retains production
    from every irradiation placement.

Both scripts tally neutron and photon flux on an identical, stationary
:class:`openmc.RegularMesh`, with dimensions ``(60, 40, 1)`` and a finite
4 cm thick z slab. They export neutron-irradiation and decay-photon-cooling
GIFs, PNG previews, and compressed NumPy arrays with times and positions.
Matplotlib's Pillow writer is used for GIF output; Pillow is needed only for
the examples, not the D1S API. The example README documents inputs and options.

The photon flux in :class:`~openmc.deplete.d1s.Results` is already physically
scaled. Divide each voxel's track-length tally by its volume to obtain
photons/cm\ :sup:`2`/s, without applying the photon rate again. For the
examples' filter ordering (mesh, photon particle), the x-y image is::

        tally = results.get_tally('decay photon mesh flux', step_name, local_time)
        mesh = tally.find_filter(openmc.MeshFilter).mesh
        data = tally.get_reshaped_data(expand_dims=True)
        flux_xy = data[:, :, 0, 0, 0, 0] / mesh.volumes[:, :, 0]
        image.set_data(flux_xy.T)  # Matplotlib rows are y, columns are x

Use a fixed color normalization across frames so intensity changes remain
visible. These are slab-volume-averaged fields at static configurations, not
continuous-motion particle tracking or a trajectory-integrated dose. Photon
error bars remain conditional on the finite captured neutron production banks.

Removal, placement, and repeated instances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once an irradiated component has entered the history, omitting its key from a
later configuration is an error. Explicitly map it to ``None`` to remove its
emission in that configuration, for example
``components={'target': None}``. Its history continues to decay and is retained
if the same physical component is reinserted later. A replacement object needs
a new identity. ``components=None`` instead means all-sources,
global-coordinate mode, with no component assignment or transforms. It can
only be combined with other no-map configurations; it is not equivalent to
removing a component.

Each transform is a finite homogeneous 4x4 rigid matrix mapping a component's
reference coordinates into that configuration's global coordinates in cm.
The last row is ``[0, 0, 0, 1]`` and the rotation must be orthonormal with
determinant +1; scaling, shearing, and reflections are rejected. Missing
transform entries mean identity. Transfer applies
``T_target @ inv(T_origin)`` to positions and only its rotation to directions.
It does not move the geometry or resample positions or angles. Supply geometry
with matching placements; transferred births are checked against destination
selectors before replay.

Each transport calculation uses a static geometry snapshot. Configuration
changes are discrete events on the irradiation/cooling timeline, not continuous
motion during particle tracking. At an observation, photons start at the
transferred birth sites and are tracked through the complete selected photon
model, using that model's materials, surfaces, and boundary conditions. Added
shielding changes attenuation and scattering even without a component source
identity. Removing a component from the source map does not edit the geometry:
to remove its attenuation too, supply a photon model without that component.
Cooling time changes the decay weights, not the photon flight time.

For repeated CSG cells, ``d1s.Cells([cell, cell], instances=[0, 2])`` selects
two zero-based occurrences. Instance numbering follows the backend's canonical
XML traversal, with cells ordered by ascending ID within each CSG universe,
not the order in which cells were inserted into the Python model. D1S applies
this ordering to model clones without reordering caller models. ``None`` for an
instance, or omission of ``instances``, selects every occurrence. Overlapping
selections are rejected. Capture membership uses the original birth's cell
instance; the target selector independently specifies allowed destination
instances. A selector alone does not relocate a birth. Use separate component
identities and transforms when occurrences need different motions.

DAGMC volume selectors
~~~~~~~~~~~~~~~~~~~~~~

:class:`~openmc.deplete.d1s.DAGMCVolumes` uses raw H5M ``GLOBAL_ID`` values for
three-dimensional volume entities, scoped to the actual
:class:`openmc.DAGMCUniverse` in the model. These are **not** runtime OpenMC
cell IDs, including IDs assigned by ``auto_geom_ids``. For example, if
``cad_a`` and ``cad_b`` are universes already placed in ``cad_model``, volume 7
in each is a different selection::

    cad_configuration = d1s.Configuration(cad_model, components={
        'insert-a': d1s.DAGMCVolumes(cad_a, [7, 7], instances=[0, 2]),
        'insert-b': d1s.DAGMCVolumes(cad_b, [7]),
    })

The first selector requires at least three placements of ``cad_a``; instance
numbers refer to zero-based volume occurrences, not raw IDs or material IDs.
As with CSG, account for every emitting region and ensure target placements
agree with any component transforms. Raw IDs are checked against the H5M tags.
Nonzero replay into a model containing DAGMC additionally runs destination
containment/instance preflight in an isolated geometry-only Python C API
subprocess. A zero-source observation in a model containing DAGMC also uses
this locator when its tallies contain a :class:`openmc.DistribcellFilter`, to
obtain backend cell-instance counts and size the zero-valued tallies correctly.
Both cases require a compiled DAGMC-capable shared library available to
``openmc.lib``, even when the photon rate is zero and no transport is run.
Zero-source observations with unstructured-mesh tallies likewise use an
isolated geometry-only subprocess to obtain mesh element counts and volumes.
They require a shared library with the selected unstructured-mesh backend,
even for CSG-only targets. Zero-source mesh results contain bin sizes and
volumes, not the vertex/connectivity data of a transport statepoint; mesh
geometry post-processing therefore still requires a real statepoint.

The locator initializes only geometry, not sources or transport, and does not
require nuclear cross-section or decay-chain data. Nonzero transport also
requires a DAGMC-capable executable. CSG-only targets without unstructured-mesh
tallies determine instance counts in Python without this subprocess;
``validate()`` does not run the locator. Both photon replay and the locator
clear any retained ``surf_source_read`` setting to prevent an extra external
source from being appended.

Normalization, uncertainty, and execution limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each captured site, its birth weight is divided by the global number of
sampled original external neutron histories and multiplied by the cumulative
parent factor. Each rank persists its count as the scalar signed 64-bit integer
``source_histories``, even when its photon bank is empty. These counts are
summed over **all** rank shards into
``Production.kernels[configuration].source_histories``, the production
normalization denominator. The count includes zero-weight histories rejected
with the ``kill`` strategy, but excludes secondary and split tracks. This
preserves the source rejection probability and matches fixed-source
normalization per initial source history and ``Irradiation.source_rate``.
Nonunit source sampling-importance weights remain in the birth numerator;
``source_weight`` is the sum of actual original source weights for information
only. Capture rejects sampled non-neutron external sites even when their weight
is zero. The sum of these physical emission weights, after component removal,
is the observation's total photon rate :math:`Q_\gamma` in photons/s. Replay
samples sites in proportion to those weights and launches unit-weight photons.
An ordinary source bank's compound layout is retained, with separate parent
tags and optional sampling weights; see :ref:`io_source_d1s` for filenames,
normalization, and metadata.

The per-birth ``history_id`` records the emitting transport track's particle ID
(``p.id()``). For shared-bank secondary tracks, this need not be the original
external-source history ID, so it cannot guarantee original-history ancestry.
This provenance limitation does not affect the independently accumulated
``source_histories`` count or production normalization.

The physical tally response is the response per replay photon multiplied
by :math:`Q_\gamma` **once**. Other score/filter units are retained: a
volume-integrated flux score becomes particle-cm/s, and still needs division
by volume to obtain particles/cm\ :sup:`2`/s. No dose conversion or volume
normalization is implicit. :meth:`~openmc.deplete.d1s.Results.get_tally`
returns an independent in-memory tally with ``d1s_metadata``. The workflow does
not insert, remove, or sum a :class:`openmc.ParentNuclideFilter`; add one
explicitly if parent-resolved results are wanted. Replay uses a unit-strength
``FileSource``; the Python workflow scales independent in-memory tallies by
:math:`Q_\gamma` exactly once. Raw statepoint files are never rescaled or
rewritten and retain their per-replay-photon normalization.

Reported standard deviations describe photon transport uncertainty conditional
on the sampled neutron birth banks. They exclude uncertainty in neutron
production/bank generation and do not provide full coupled neutron/photon
uncertainty. An observation with zero photon rate skips transport, has no
statepoint or replay-bank path, and returns zero-valued tallies; a zero sampled
bank is not proof of zero physical activation.

Capture/replay supports only fixed-source, continuous-energy Monte Carlo, not
eigenvalue, multigroup, random-ray, statepoint-restart, or particle-restart
workflows. Nonzero transport requires an OpenMC executable with the D1S capture
and weighted replay backend; the Python API alone is insufficient. Transport
runs in executable subprocesses, not an initialized caller C API session. If
``openmc.lib`` is initialized, finalize it explicitly before this workflow; its
state is not changed on rejection. ``run_kwargs`` accepts only ``particles``,
``threads``, ``geometry_debug``, ``tracks``, ``output``, ``openmc_exec``,
``mpi_args``, and ``event_based``. Input/output locations, exports, restarts,
and updates to caller models cannot be overridden through these options.
