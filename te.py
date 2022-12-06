import openmc
import openmc.deplete
from openmc.data import decay_photon_energy

m = openmc.Material()
m.add_nuclide('I135', 1.0e-24)
m.add_nuclide('Cs135', 1.0e-24)
m.volume = 1.0
src = m.decay_photon_energy()
