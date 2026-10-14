.. _pythonapi_deplete:

.. module:: openmc.deplete

----------------------------------
:mod:`openmc.deplete` -- Depletion
----------------------------------

Primary API
-----------

The two primary requirements to perform depletion with :mod:`openmc.deplete`
are:

    1) A transport operator
    2) A time-integration scheme

The former is responsible for calculating and retaining important information
required for depletion. The most common examples are reaction rates and power
normalization data. The latter is responsible for projecting reaction rates and
compositions forward in calendar time across some step size :math:`\Delta t`,
and obtaining new compositions given a power or power density. The
:class:`CoupledOperator` class is provided to obtain reaction rates via tallies
through OpenMC's transport solver, and the :class:`IndependentOperator` class is
provided to obtain reaction rates from cross-section data. Several classes are
provided that implement different time-integration algorithms for depletion
calculations, which are described in detail in Colin Josey's thesis,
`Development and analysis of high order neutron transport-depletion coupling
algorithms <https://dspace.mit.edu/handle/1721.1/113721>`_.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template:  myintegrator.rst

    PredictorIntegrator
    CECMIntegrator
    CELIIntegrator
    CF4Integrator
    EPCRK4Integrator
    LEQIIntegrator
    SICELIIntegrator
    SILEQIIntegrator

Each of these classes expects a "transport operator" to be passed. OpenMC
provides the following transport operator classes:

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: mycallable.rst

   CoupledOperator
   IndependentOperator

The :class:`CoupledOperator` and :class:`IndependentOperator` classes must also
have some knowledge of how nuclides transmute and decay. This is handled by the
:class:`Chain` class.

The :class:`IndependentOperator` class requires a set of fluxes and microscopic
cross sections. The following function can be used to generate this information:

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myfunction.rst

   get_microxs_and_flux

Minimal Example
---------------

A minimal example for performing depletion would be:

.. code::

    >>> import openmc
    >>> import openmc.deplete
    >>> geometry = openmc.Geometry.from_xml()
    >>> settings = openmc.Settings.from_xml()
    >>> model = openmc.Model(geometry, settings)

    # Representation of a depletion chain
    >>> chain_file = "chain_casl.xml"
    >>> operator = openmc.deplete.CoupledOperator(model, chain_file)

    # Set up 5 time steps of one day each
    >>> dt = [24 * 60 * 60] * 5
    >>> power = 1e6  # constant power of 1 MW

    # Deplete using mid-point predictor-corrector
    >>> cecm = openmc.deplete.CECMIntegrator(operator, dt, power)
    >>> cecm.integrate()

Internal Classes and Functions
------------------------------

When running in parallel using `mpi4py
<https://mpi4py.readthedocs.io/en/stable/>`_, the MPI intercommunicator used can
be changed by modifying the following module variable. If it is not explicitly
modified, it defaults to ``mpi4py.MPI.COMM_WORLD``.

.. data:: comm

   MPI intercommunicator used to call OpenMC library

   :type: mpi4py.MPI.Comm

During a depletion calculation, the depletion chain, reaction rates, and number
densities are managed through a series of internal classes that are not normally
visible to a user. However, should you find yourself wondering about these
classes (e.g., if you want to know what decay modes or reactions are present in
a depletion chain), they are documented here. The following classes store data
for a depletion chain:

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myclass.rst

   Chain
   DecayTuple
   Nuclide
   ReactionTuple
   FissionYieldDistribution
   FissionYield

The :class:`Chain` class uses information from the following module variable:

.. data:: chain.REACTIONS

   Dictionary that maps transmutation reaction names to information needed when
   a chain is being generated: MT values, the change in atomic/mass numbers
   resulting from the reaction, and what secondaries are produced.

   :type: dict

The following classes are used during a depletion simulation and store auxiliary
data, such as number densities and reaction rates for each material.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myclass.rst

   AtomNumber
   MicroXS
   OperatorResult
   ReactionRates
   Results
   StepResult

The following class and functions are used to solve the depletion equations,
with :func:`cram.CRAM48` being the default.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myintegrator.rst

   cram.IPFCramSolver

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myfunction.rst

   cram.CRAM16
   cram.CRAM48
   pool.deplete

.. data:: pool.USE_MULTIPROCESSING

   Boolean switch to enable or disable the use of :mod:`multiprocessing`
   when solving the Bateman equations. The default is to use
   :mod:`multiprocessing`, but can cause the simulation to hang in
   some computing environments, namely due to MPI and networking
   restrictions. Disabling this option will result in only a single
   CPU core being used for depletion.

   :type: bool

.. data:: pool.NUM_PROCESSES

   Number of worker processes used for depletion calculations, which rely on the
   :class:`multiprocessing.pool.Pool` class. If set to ``None`` (default), the
   number returned by :func:`os.cpu_count` is used.

The following classes are used to help the :class:`openmc.deplete.CoupledOperator`
compute quantities like effective fission yields, reaction rates, and
total system energy.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myclass.rst

   helpers.AveragedFissionYieldHelper
   helpers.ChainFissionHelper
   helpers.ConstantFissionYieldHelper
   helpers.DirectReactionRateHelper
   helpers.EnergyScoreHelper
   helpers.FissionYieldCutoffHelper
   helpers.FluxCollapseHelper

The :class:`openmc.deplete.IndependentOperator` uses inner classes subclassed
from those listed above to perform similar calculations.

The following classes are used to define external source rates or transfer rates
to model continuous removal or feed of nuclides during depletion.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myclass.rst

   transfer_rates.ExternalSourceRates
   transfer_rates.TransferRates

Intermediate Classes
--------------------

Specific implementations of abstract base classes may utilize some of
the same methods and data structures. These methods and data are stored
in intermediate classes.

Methods common to tally-based implementation of :class:`FissionYieldHelper`
are stored in :class:`helpers.TalliedFissionYieldHelper`

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myclass.rst

   helpers.TalliedFissionYieldHelper

Methods common to OpenMC-specific implementations of :class:`TransportOperator`
are stored in :class:`openmc_operator.OpenMCOperator`

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: mycallable.rst

   openmc_operator.OpenMCOperator


Abstract Base Classes
---------------------

A good starting point for extending capabilities in :mod:`openmc.deplete` is
to examine the following abstract base classes. Custom classes can
inherit from :class:`abc.TransportOperator` to implement alternative
schemes for collecting reaction rates and other data prior to depleting
materials

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: mycallable.rst

   abc.TransportOperator

The following classes are abstract classes used to pass information from
transport simulations (in the case of transport-coupled depletion) or to
simply calculate these quantities directly (in the case of
transport-independent depletion) back on to the :class:`abc.TransportOperator`

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myclass.rst

   abc.NormalizationHelper
   abc.FissionYieldHelper
   abc.ReactionRateHelper

Custom integrators or depletion solvers can be developed by subclassing from
the following abstract base classes:

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myintegrator.rst

   abc.Integrator
   abc.SIIntegrator
   abc.DepSystemSolver

R2S Automation
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myclass.rst

   R2SManager

D1S Capture and Replay
----------------------

.. module:: openmc.deplete.d1s

.. currentmodule:: openmc.deplete

The :mod:`openmc.deplete.d1s` interface weights reaction-produced decay-photon
births for an irradiation/cooling schedule. Despite its location in the
depletion package, it does not evolve compositions or calculate a depletion
inventory. See :ref:`usersguide_d1s_multi_model` for a staged example, physical
assumptions, component mapping, backend requirements, normalization, and
uncertainty.

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: myclass.rst

    d1s.Cells
    d1s.DAGMCVolumes
    d1s.Configuration
    d1s.Irradiation
    d1s.Cooling
    d1s.Simulation
    d1s.Production
    d1s.Observation
    d1s.Results

The main entry points are:

* :meth:`d1s.Simulation.validate` checks the schedule, selectors, and model
  settings without transport. It does not verify executable backend support.
* ``Simulation.run(output_dir='d1s', run_kwargs=None)`` prepares production and
  evaluates the schedule, returning :class:`d1s.Results`.
* ``Simulation.prepare(output_dir='d1s-production', run_kwargs=None)`` returns
  :class:`d1s.Production` with one captured bank per distinct positive-rate
  irradiation :class:`d1s.Configuration` object.
* :meth:`d1s.Simulation.evaluate` reuses prepared in-process production.
  Irradiation configuration objects and the decay-data file must match.
  Rates, durations, cooling models, and sample times may change. Prepared
  configurations and their models must remain unchanged.
* ``Results.get_tally(name, step, time)`` returns an independent tally copy.
  ``name`` identifies a unique tally by name or ID. ``step`` is a step name
  or zero-based schedule index. ``time`` must match a requested local sample
  offset in seconds or as a ``(value, unit)`` pair.

``Production.kernels[configuration].source_histories`` is the production
normalization denominator: the total number of sampled original external
neutron histories across all rank shards, including zero-weight histories
rejected with the ``kill`` strategy and ranks with empty photon banks.
Secondary and split tracks are excluded. The kernel's ``source_weight`` is the
sum of the actual original source weights, for information only. Birth weights
retain source sampling-importance corrections. This matches fixed-source
normalization per initial source history and the initial neutron rate supplied
by ``Irradiation.source_rate``. See :ref:`io_source_d1s` for the persisted
scalars.

Per-birth ``history_id`` metadata records the emitting track's particle ID
(``p.id()``), not guaranteed original external-source ancestry when secondary
tracks use a shared bank. This provenance limitation does not affect the
independently accumulated ``source_histories`` count or normalization.

All output directories must be new or empty. ``run_kwargs`` accepts only
``particles``, ``threads``, ``geometry_debug``, ``tracks``, ``output``,
``openmc_exec``, ``mpi_args``, and ``event_based``. The workflow requires
fixed-source continuous-energy Monte Carlo and an uninitialized caller C API;
production accepts only neutron external sources, not mixed neutron/photon
sources. Nonzero transport needs a compatible executable. For targets
containing DAGMC, nonzero replay preflight and zero-source
:class:`openmc.DistribcellFilter` sizing use isolated geometry-only Python
C API subprocesses. Zero-source unstructured-mesh tally sizing also uses an
isolated geometry-only subprocess to obtain element counts and volumes,
including in CSG-only targets. The shared library must support DAGMC or the
selected mesh backend, respectively, even when the photon rate is zero. These
subprocesses initialize neither sources nor transport and require no nuclear
cross-section or decay-chain data. Both photon replay and the locator clear
retained ``surf_source_read`` settings to prevent an extra external source.

``Results.observations[(step_name, local_seconds)]`` provides an
:class:`d1s.Observation`, including cumulative time, photon source rate in
photons/s, replay-bank and raw-statepoint paths, tallies, and provenance.
``Results.source_rates`` and ``Results.statepoint_paths`` use the same keys.
Zero-source observations have zero tallies and ``None`` paths. Replay uses a
unit-strength ``FileSource``, and the workflow multiplies its per-photon tally
response by the physical photon rate exactly once. Raw statepoint files are not
modified. Standard deviations include only photon transport uncertainty
conditional on the captured banks, not neutron production/bank-generation
uncertainty.

D1S Helper Functions
--------------------

These lower-level helpers support time correction of parent-resolved tallies
from a single transport configuration. They are not additional processing
steps for :class:`d1s.Simulation` results.

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: myfunction.rst

   d1s.get_radionuclides
   d1s.prepare_tallies
   d1s.time_correction_factors
   d1s.apply_time_correction
