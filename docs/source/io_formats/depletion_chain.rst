.. _io_depletion_chain:

============================
Depletion Chain -- chain.xml
============================

A depletion chain file has a ``<depletion_chain>`` root element with one or more
``<nuclide>`` child elements. The decay, reaction, and fission product data for
each nuclide appears as child elements of ``<nuclide>``.

---------------------
``<nuclide>`` Element
---------------------

The ``<nuclide>`` element contains information on the decay modes, reactions,
and fission product yields for a given nuclide in the depletion chain. This
element may have the following attributes:

  :name:
    Name of the nuclide

  :half_life:
    Half-life of the nuclide in [s]

  :decay_modes:
    Number of decay modes present

  :decay_energy:
    Decay energy released in [eV]

  :reactions:
    Number of reactions present

For each decay mode, a :ref:`io_chain_decay` appears as a child of
``<nuclide>``. For each reaction present, a :ref:`io_chain_reaction` appears as
a child of ``<nuclide>``. If the nuclide is fissionable, a :ref:`io_chain_nfy`
appears as well.

.. _io_chain_decay:

-------------------
``<decay>`` Element
-------------------

The ``<decay>`` element represents a single decay mode and has the following
attributes:

  :type:
    The type of the decay, e.g. 'ec/beta+'

  :target:
    The daughter nuclide produced from the decay

  :branching_ratio:
    The branching ratio for this decay mode

.. _io_chain_reaction:

--------------------
``<source>`` Element
--------------------

The ``<source>`` element represents photon and electron sources associated with
the decay of a nuclide and contains information to construct an
:class:`openmc.stats.Univariate` object that represents this emission as an
energy distribution. This element has the following attributes:

  :type:
    The type of :class:`openmc.stats.Univariate` source term.

  :particle:
    The type of particle emitted, e.g., 'photon' or 'electron'

  :parameters:
    The parameters of the source term, e.g., for a
    :class:`openmc.stats.Discrete` source, the energies (in [eV]) at which the
    particles are emitted and their relative intensities in [Bq/atom] (in other
    words, decay constants).

----------------------
``<reaction>`` Element
----------------------

The ``<reaction>`` element represents a single transmutation reaction. This
element has the following attributes:

  :type:
    The type of the reaction, e.g., '(n,gamma)'

  :Q:
    The Q value of the reaction in [eV]

  :target:
    The nuclide produced in the reaction (absent if the type is 'fission')

  :branching_ratio:
    The branching ratio for the reaction

In addition to these attributes, a ``<reaction>`` element may contain zero or
more :ref:`io_chain_isomeric` sub-elements carrying verbatim energy-dependent
isomeric production data from the source evaluation.

.. _io_chain_isomeric:

---------------------------------
``<isomeric_production>`` Element
---------------------------------

.. versionadded:: 0.15.4

The ``<isomeric_production>`` element stores energy-dependent isomeric
production data (ENDF MF=9 yields and/or MF=10 partial cross sections) for the
final state produced by the parent ``<reaction>`` element, exactly as given in
the source evaluation. The scalar ``branching_ratio`` attributes on
``<reaction>`` elements remain authoritative for consumers that do not use
this data, and the scalar branching ratios of a reaction type always sum to
about one. Note that the stored functions are verbatim evaluation data: an
MF=10 partial cross section requires an external total cross section to form a
branching ratio, and the per-state MF=9 yields of a reaction do not
necessarily sum to one when the ground-state share is implicit.

A ``<reaction>`` element may hold several ``<isomeric_production>`` elements,
e.g. when a final state that could not be matched to a known metastable state
is folded onto the ground-state target. Consumers should sum the tables of all
instances attached to one target; OpenMC never performs arithmetic on the
stored data.

This element has the following attributes:

  :level:
    ENDF LFS level number of the final state, verbatim from the source
    evaluation. This is a level index, not a metastable-state index.

  :excitation_energy:
    Excitation energy of the final state in [eV] (the MF=8 ELFS value when
    present, otherwise QM minus QI). Zero for the ground state.

Each ``<isomeric_production>`` element contains one or more ``<table>``
sub-elements, each holding one tabulated function with the following
attributes and sub-elements:

  :mf:
    ENDF file number the data comes from: 9 for energy-dependent yields
    (dimensionless multiplicities of the reaction cross section) or 10 for
    partial production cross sections in [b]

  :mt:
    ENDF reaction number of the section that supplied the data

  :source:
    Source library identifier as recorded in the evaluation header,
    e.g. 'ENDF/B-8.1' or 'TENDL-2023.1'

  :QM:
    Mass-difference Q value in [eV], verbatim from the TAB1 header

  :QI:
    Reaction Q value for this particular state in [eV], verbatim from the
    TAB1 header

  :breakpoints:
    Whitespace-separated breakpoints of the interpolation regions

  :interpolation:
    Whitespace-separated ENDF interpolation scheme codes for each region,
    following the same convention as :class:`openmc.data.Tabulated1D`

  :energies:
    Sub-element listing the incident neutron energies in [eV]

  :values:
    Sub-element listing the tabulated yields or cross sections

Only ENDF "scheme 1" isomeric data (MF=8/9/10) is represented in the chain
file. Evaluations that encode isomer production via MF=6 product distributions
or via discrete-level reaction sections are not captured.

.. _io_chain_nfy:

------------------------------------
``<neutron_fission_yields>`` Element
------------------------------------

The ``<neutron_fission_yields>`` element provides yields of fission products for
fissionable nuclides. Normally, it has the follow sub-elements:

  :energies:
    Energies in [eV] at which yields for products are tabulated

  :fission_yields:

    Fission product yields for a single energy point. This element itself has a
    number of attributes/sub-elements:

      :energy:
        Energy in [eV] at which yields are tabulated

      :products:
        Names of fission products

      :data:
        Independent yields for each fission product

In the event that a nuclide doesn't have any known fission product yields, it is
possible to have that nuclide borrow yields from another nuclide by indicating
the other nuclide in a single `parent` attribute. For example:

.. code-block:: xml

    <neutron_fission_yields parent="U235"/>
