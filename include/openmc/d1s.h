//! \file d1s.h
//! \brief Capture sampled decay-photon births for transport in another model.

#ifndef OPENMC_D1S_H
#define OPENMC_D1S_H

#include "openmc/position.h"

namespace openmc {

class Particle;

//! Check capture settings, including unsupported restart modes.
void validate_d1s_settings();

//! Reset rank-local capture state at simulation initialization.
void reset_d1s_source();

//! Count an initial neutron history and add its original source weight.
//!
//! Include zero-weight, kill-rejected histories; exclude all secondary tracks.
//! The history count is the normalization denominator, not the weight sum.
void accumulate_d1s_source(double weight);

//! Capture a sampled birth, before secondary creation or transport cutoffs.
//!
//! Positions/directions are global. Only the emitting leaf cell is identified
//! in schema v1; ancestor/component coordinate mapping is left to the caller.
//! Storage is O(number of births on this rank), with no truncation or dropping.
void bank_d1s_photon(
  const Particle& p, Direction u, double E, double weight, int parent_nuclide);

//! Write this rank's bank using serial HDF5, including an empty bank.
void write_d1s_source();

//! Release capture storage during library finalization.
void free_memory_d1s();

} // namespace openmc

#endif // OPENMC_D1S_H