#include "openmc/d1s.h"

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <type_traits>

#include <fmt/core.h>

#include "openmc/cell.h"
#include "openmc/chain.h"
#include "openmc/constants.h"
#include "openmc/dagmc.h"
#include "openmc/error.h"
#include "openmc/hdf5_interface.h"
#include "openmc/message_passing.h"
#include "openmc/particle.h"
#include "openmc/settings.h"
#include "openmc/simulation.h"
#include "openmc/state_point.h"
#include "openmc/universe.h"
#include "openmc/vector.h"

namespace openmc {
namespace {

struct Birth {
  SourceSite site;
  int cell_id;
  int cell_instance;
  int universe_id;
  int dagmc_volume_id;
  int64_t history_id;
  int batch;
  int collision;
};

vector<Birth> births;
int64_t source_histories {0};
double source_weight {0.0};
std::mutex bank_mutex;

// Resolve ties beyond the collision number by phase space, not thread arrival
// order. Different secondary tracks in one history can share collision numbers.
auto sort_key(const Birth& b)
{
  const auto& s = b.site;
  return std::make_tuple(b.batch, b.history_id, b.collision, b.cell_id,
    b.cell_instance, b.universe_id, b.dagmc_volume_id, s.parent_nuclide, s.r.x,
    s.r.y, s.r.z, s.u.x, s.u.y, s.u.z, s.E, s.wgt);
}

} // namespace

void validate_d1s_settings()
{
  if (!settings::d1s_source_write)
    return;

  if (!settings::use_decay_photons || !settings::photon_transport ||
      !settings::run_CE || settings::run_mode != RunMode::FIXED_SOURCE ||
      settings::solver_type != SolverType::MONTE_CARLO) {
    fatal_error(
      "d1s_source_write requires use_decay_photons, photon_transport, "
      "and continuous-energy fixed-source Monte Carlo transport.");
  }
  if (settings::restart_run || settings::particle_restart_run) {
    fatal_error("d1s_source_write does not support statepoint or particle "
                "restart capture.");
  }
}

void reset_d1s_source()
{
  births.clear();
  source_histories = 0;
  source_weight = 0.0;
}

void accumulate_d1s_source(double weight)
{
  std::lock_guard<std::mutex> lock {bank_mutex};
  ++source_histories;
  source_weight += weight;
}

void bank_d1s_photon(
  const Particle& p, Direction u, double E, double weight, int parent_nuclide)
{
  if (parent_nuclide < 0 || parent_nuclide >= data::chain_nuclides.size()) {
    fatal_error("D1S photon birth has an invalid decay parent nuclide.");
  }

  const auto& coord = p.lowest_coord();
  const auto& cell = *model::cells[coord.cell()];
  const auto& universe = *model::universes[coord.universe()];
  Birth birth {};
  birth.site.r = p.r();
  birth.site.u = u;
  birth.site.E = E;
  birth.site.wgt = weight;
  birth.site.time = 0.0;
  birth.site.surf_id = SURFACE_NONE;
  birth.site.particle = ParticleType::photon();
  birth.site.parent_nuclide = parent_nuclide;
  birth.cell_id = cell.id_;
  birth.cell_instance = p.cell_instance();
  birth.universe_id = universe.id_;
  birth.history_id = p.id();
  birth.batch = simulation::current_batch;
  birth.collision = p.n_collision();

#ifdef OPENMC_DAGMC_ENABLED
  if (const auto* dag_cell = dynamic_cast<const DAGCell*>(&cell)) {
    const auto* dag_universe = dynamic_cast<const DAGUniverse*>(&universe);
    if (!dag_universe)
      fatal_error("D1S DAGMC birth is not in a DAGMC universe.");
    birth.dagmc_volume_id =
      dag_universe->dagmc_ptr()->id_by_index(3, dag_cell->dag_index());
  }
#endif

  std::lock_guard<std::mutex> lock {bank_mutex};
  births.push_back(birth);
}

void write_d1s_source()
{
  // Called after transport has finished. No MPI communication is used here:
  // every rank opens a distinct file with the serial HDF5 driver, even in a
  // PHDF5 build. In particular, do not use write_source_bank here.
  std::sort(births.begin(), births.end(),
    [](const Birth& a, const Birth& b) { return sort_key(a) < sort_key(b); });
  auto path =
    settings::path_output + fmt::format("d1s_source.{}.h5", mpi::rank);
  hid_t file = file_open(path, 'w', false);
  write_attribute(file, "filetype", "source");
  write_attribute(file, "version", VERSION_STATEPOINT);
  write_attribute(file, "d1s_version", 1);

  // The on-disk compound is exactly the ordinary source-bank compound. Parent
  // tags and geometry metadata are separate, aligned one-dimensional arrays.
  hsize_t n = births.size();
  hid_t space = H5Screate_simple(1, &n, nullptr);
  hid_t file_type = h5banktype(false);
  hid_t memory_type = h5banktype(true);
  hid_t dataset = H5Dcreate2(file, "source_bank", file_type, space, H5P_DEFAULT,
    H5P_DEFAULT, H5P_DEFAULT);
  if (dataset < 0)
    fatal_error("Could not create D1S source_bank dataset.");
  if (n > 0) {
    vector<SourceSite> sites;
    sites.reserve(n);
    for (const auto& birth : births)
      sites.push_back(birth.site);
    if (H5Dwrite(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT,
          sites.data()) < 0)
      fatal_error("Could not write D1S source_bank dataset.");
  }
  H5Dclose(dataset);
  H5Tclose(memory_type);
  H5Tclose(file_type);
  H5Sclose(space);

  vector<std::string> names;
  names.reserve(data::chain_nuclides.size());
  for (const auto& nuclide : data::chain_nuclides)
    names.push_back(nuclide->name());
  write_dataset(file, "parent_names", names);

  vector<int> parents;
  parents.reserve(n);
  for (const auto& birth : births)
    parents.push_back(birth.site.parent_nuclide);
  write_dataset(file, "parent_nuclide", parents);

  auto write_member = [&](const char* name, auto member) {
    using T = std::decay_t<decltype(Birth {}.*member)>;
    vector<T> values;
    values.reserve(n);
    for (const auto& birth : births)
      values.push_back(birth.*member);
    write_dataset(file, name, values);
  };
  write_member("cell_id", &Birth::cell_id);
  write_member("cell_instance", &Birth::cell_instance);
  write_member("universe_id", &Birth::universe_id);
  write_member("dagmc_volume_id", &Birth::dagmc_volume_id);
  write_member("history_id", &Birth::history_id);
  write_member("batch", &Birth::batch);
  write_dataset(file, "source_histories", source_histories);
  write_dataset(file, "source_weight", source_weight);
  file_close(file);
}

void free_memory_d1s()
{
  vector<Birth>().swap(births);
  source_histories = 0;
  source_weight = 0.0;
}

} // namespace openmc