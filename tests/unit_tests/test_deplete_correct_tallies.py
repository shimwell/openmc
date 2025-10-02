
import openmc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
import numpy as np
import argparse
from openmc.deplete import d1s

from pathlib import Path

# multiplication by pico_to_milli converts from (pico) pSv to (milli) mSv
pico_to_milli = 1e-9
seconds_to_hours = 3600
openmc.config['chain_file'] = Path.home() / 'nuclear_data' / 'chain_endf_b8.0.xml'

original_model=openmc.Model().from_model_xml('/home/jon/neutronics/original_model.xml') # 13 seconds just to load the model including the source

statepoint_filename = "/home/jon/neutronics/stuff10/statepoint_shutdown_dose_dt.h5"

# Get tally from DT simulation statepoint
with openmc.StatePoint(statepoint_filename) as sp:
    dose_tally_from_sp_dt = sp.get_tally(name='photon_dose_on_mesh')


# this gets all the unstable nuclides that can be produced during D1S
# both models have the same materials so this function will return the same radionuclides regarless of which model (model_dt or model dd) is passed in.
radionuclides = d1s.get_radionuclides(original_model)

months_to_safe = []
neutron_budgets = []

# for neutron_budget in :
timesteps = [1]
source_rates_dt = [1e18]
for month in range(1, 12*15):
    timesteps.append((365.25*24*60*60)/12)  # 1 month in seconds
    source_rates_dt.append(0)  # no source during cooling
# Compute time correction factors based on irradiation schedule
import time

# Time the original function
start_time = time.time()
time_factors_dt = d1s.time_correction_factors(
    nuclides=radionuclides,
    timesteps=timesteps,
    source_rates=source_rates_dt,
    timestep_units = 's'
)
original_time = time.time() - start_time
print(f"Original function time: {original_time:.4f} seconds")

# Time the vectorized function
start_time = time.time()
time_factors_dt_vec = d1s.time_correction_factors_vectorized(
    nuclides=radionuclides,
    timesteps=timesteps,
    source_rates=source_rates_dt,
    timestep_units='s'
)
vectorized_time = time.time() - start_time
print(f"Vectorized function time: {vectorized_time:.4f} seconds")
print(f"Speedup: {original_time/vectorized_time:.2f}x")

# Check if results are identical
Co60_diff = np.max(np.abs(time_factors_dt['Co60'] - time_factors_dt_vec['Co60']))
print(f"Max difference in Co60 results: {Co60_diff:.2e}")

# Check all nuclides for differences
max_diff_overall = 0
for nuc in radionuclides:
    diff = np.max(np.abs(time_factors_dt[nuc] - time_factors_dt_vec[nuc]))
    max_diff_overall = max(max_diff_overall, diff)
print(f"Max difference across all nuclides: {max_diff_overall:.2e}")

