# Moving-source and moving-volume D1S examples

Both examples use a **stationary `openmc.RegularMesh`**, not an unstructured
mesh. They create a neutron-flux GIF for the irradiation intervals and a
decay-photon-flux GIF for cooling observations.

## Requirements

- Build and install the D1S capture/replay branch. Set `OPENMC_CROSS_SECTIONS`
  to continuous-energy nickel and hydrogen neutron/photon data.
- Set `OPENMC_CHAIN_FILE` to a compatible reaction/decay chain containing
  nickel activation products, or pass `--chain-file`. The small nickel test
  chain in this repository can be used for demonstration, not as a complete
  production activation library.
- NumPy, Matplotlib, and Pillow are used for post-processing. Pillow is only
  needed for GIF export. No ffmpeg or VTK installation is required.
- Leave any caller `openmc.lib` session uninitialized. Use new or empty output
  directories. These scripts never overwrite existing simulation directories.

Run either script from any directory, with the compiled executable on PATH:

```sh
python moving_neutron_source.py --output moving-source-results
python moving_activated_volume.py --output moving-volume-results
```

If running from the repository root, prepend `examples/d1s_moving/` to each
script name. Use `--openmc-exec` to select a different compiled executable.
Other options are `--particles`, `--batches`, `--mesh-bins NX NY`, `--threads`,
and `--fps`. Increase particle counts and check convergence before using the
results for an engineering assessment.

## Example 1: everything fixed except the neutron source

[moving_neutron_source.py](moving_neutron_source.py) holds the nickel volume,
room, hydrogen shielding slab, and tally mesh fixed. Only the neutron source
changes between three static irradiation snapshots:

| Irradiation interval (min) | Neutron source position (cm) | Nickel center (cm) |
| --- | --- | --- |
| 0 to 20 | (-12, -4, 0) | (0, 0, 0) |
| 20 to 40 | (-12, 0, 0) | (0, 0, 0) |
| 40 to 60 | (-12, 4, 0) | (0, 0, 0) |

The source is a 14.1 MeV beam pointing along +x, with a physical rate of
`1.e12` initial neutrons/s. Its three spatial distributions require three
production `Configuration` objects. `neutron_flux.gif` shows these three
irradiation fields with a moving neutron-source marker.

After irradiation stops, the same stationary nickel volume emits decay
photons. Twelve cooling observations cover 0 to 60 minutes after shutdown.
`photon_flux.gif` shows their summed, time-weighted activation response. The
neutron sources are off, and **the gamma-emitting geometry does not move**.
Its spatial emission pattern can change as parents from the different
irradiation states decay at different rates.

## Example 2: fixed neutron source, one moving activation volume

[moving_activated_volume.py](moving_activated_volume.py) keeps the neutron
source at **(-15, 0, 0) cm** for every irradiation. It is independent of the
`emitter_x_cm` argument. The room, shielding slab, and mesh are stationary.
Only the nickel sphere is repositioned:

| Irradiation interval (min) | Nickel center x (cm) | Neutron source |
| --- | ---: | --- |
| 0 to 20 | 0 | Fixed at (-15, 0, 0) |
| 20 to 40 | 12 | Fixed at (-15, 0, 0) |
| 40 to 60 | 24 | Fixed at (-15, 0, 0) |

Each placement changes the irradiation geometry, so it has its own neutron
production bank. The stable component key `emitter` identifies the **same
physical nickel object** in all configurations, with cumulative activation.

Neutron irradiation then stops. During cooling the activated sphere moves
through x=24, 18, 12, 6, 0, and -6 cm in successive ten-minute intervals.
Observations at offsets 0 and 5 minutes in each interval produce twelve GIF
frames. Source points from **all three irradiation banks** are transferred
to the current nickel placement. The actual nickel geometry moves too,
while attenuation in the stationary shielding is recalculated by photon
transport. Cooling models have no assigned neutron source.

## Mesh tallies, physical units, and animation

The shared helper, [d1s_mesh_common.py](d1s_mesh_common.py), defines:

```python
mesh = openmc.RegularMesh()
mesh.dimension = (60, 40, 1)
mesh.lower_left = (-30.0, -24.0, -2.0)
mesh.upper_right = (60.0, 24.0, 2.0)

tally = openmc.Tally(name='decay photon mesh flux')
tally.filters = [openmc.MeshFilter(mesh), openmc.ParticleFilter('photon')]
tally.scores = ['flux']
```

The grid is identical in every configuration. It has one **finite-thickness
z bin**, spanning -2 to 2 cm, rather than a zero-thickness plane. Each image
is a voxel-volume-averaged flux in this slab, not a point value at z=0.

Returned D1S tallies are already scaled to the physical photon rate. The
helper divides the track-length flux by each voxel's volume to obtain
photons/cm²/s, then transposes the `(x, y)` data for Matplotlib's `(row=y,
column=x)` layout. It does not apply the photon rate again. Neutron production
statepoint tallies are instead scaled by their irradiation neutron rate and
divided by voxel volume to obtain neutrons/cm²/s.

Every frame in a GIF uses the same logarithmic color scale, so a changing
intensity is not hidden by automatic rescaling. The white outline is the
nickel volume, cyan marks the stationary H1 slab, and green crosses identify
neutron-source sites. In cooling frames, the crosses are explicitly marked
as off. Matplotlib's `PillowWriter` exports GIFs without an external video
encoder. Display playback speed is illustrative; frame titles give the
physical schedule time.

Each output directory contains:

- `neutron_flux.gif`: three static irradiation snapshots.
- `photon_flux.gif`: twelve decay-photon cooling snapshots.
- First and last PNG frames for both animations.
- `neutron_mesh_flux.npz`: physical neutron means, transport standard
  deviations, source positions, and irradiation interval times.
- `mesh_flux.npz`: physical photon means, conditional standard deviations,
  source rates, positions, and local/global cooling times. Flux arrays have
  shape `(frame, nx, ny)`.
- The D1S production banks, per-observation replay banks, and raw statepoints.

The scripts print a CSV-style summary of each cooling observation, including
time after shutdown, nickel x coordinate, photon source rate, and maximum
mesh flux, followed by the output locations. Numerical values depend on the
chain, cross sections, histories, and mesh resolution.

## Interpretation and limits

These animations represent **discrete static configurations**, not continuous
boundary movement during tracking or dose integrated along a travel path.
The neutron GIF depicts the steady field for each irradiation interval; it
is not a time-of-flight simulation. The gamma GIF depicts cooling only, not
concurrent neutron/gamma transport during irradiation. This D1S API generates
photon observations in `Cooling` steps and retains fixed-composition
production kernels, not depletion inventories.

The pure H1 slab is an illustrative non-decay-emitting moderator/shield, not
a realistic engineering shield design. If it is replaced with activating
material, give every emitting region a component selector and appropriate
history too. All sphere placements in these examples avoid overlap with it.

Photon standard deviations are conditional on the sampled neutron birth
banks and omit neutron production uncertainty. Smooth animations and low
photon-only errors do not establish converged activation. These examples
demonstrate the API and spatial fields, not a validated dose calculation.