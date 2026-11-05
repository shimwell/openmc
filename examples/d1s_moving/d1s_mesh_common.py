"""Shared geometry, RegularMesh post-processing, and animation helpers."""

import argparse
from pathlib import Path

import numpy as np
import openmc
from openmc.deplete import d1s


EMITTER_RADIUS_CM = 6.0
ROOM_LOWER_LEFT_CM = (-50.0, -35.0, -20.0)
ROOM_UPPER_RIGHT_CM = (70.0, 35.0, 20.0)
SHIELD_X_CM = (32.0, 36.0)
IRON_Y_CM = (-18.0, -12.0)
IRON_Z_CM = (-8.0, 8.0)
# Every example uses a monodirectional beam along this direction.
NEUTRON_DIRECTION = (1.0, 0.0, 0.0)
SOURCE_ARROW_CM = 8.0
MESH_LOWER_LEFT_CM = (-30.0, -24.0, -2.0)
MESH_UPPER_RIGHT_CM = (60.0, 24.0, 2.0)
TALLY_NAME = 'decay photon mesh flux'
NEUTRON_TALLY_NAME = 'irradiation neutron mesh flux'


def parse_arguments(description, default_output, cad=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--output', type=Path, default=Path(default_output),
                        help='New or empty output directory')
    parser.add_argument('--chain-file', type=Path,
                        default=openmc.config.get('chain_file'),
                        help='Reaction/decay chain (or OPENMC_CHAIN_FILE)')
    parser.add_argument('--particles', type=int, default=5000)
    parser.add_argument('--batches', type=int, default=5)
    parser.add_argument('--mesh-bins', type=int, nargs=2, default=(60, 40),
                        metavar=('NX', 'NY'))
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--openmc-exec', default='openmc')
    parser.add_argument('--fps', type=int, default=3)
    if cad:
        parser.add_argument(
            '--cad-dir', type=Path, default=Path(f'{default_output}-h5m'),
            help='Directory for DAGMC files, reused if already present')
    args = parser.parse_args()
    if args.chain_file is None:
        parser.error('Set --chain-file or OPENMC_CHAIN_FILE')
    if args.particles <= 0 or args.batches < 2:
        parser.error('Use positive particles and at least two batches')
    if min(args.mesh_bins) <= 0 or args.threads <= 0 or args.fps <= 0:
        parser.error('Mesh dimensions, threads, and fps must be positive')
    return args


def make_materials(names):
    """Return the nickel, hydrogen and iron materials with given names."""
    nickel = openmc.Material(name=names[0])
    nickel.add_element('Ni', 1.0)
    nickel.set_density('g/cm3', 8.9)

    # Pure H1 is an illustrative static moderator/shield. Its neutron-capture
    # product is stable, so this region needs no decay-photon source identity.
    hydrogen = openmc.Material(name=names[1])
    hydrogen.add_nuclide('H1', 1.0)
    hydrogen.set_density('g/cm3', 0.07)

    iron = openmc.Material(name=names[2])
    iron.add_element('Fe', 1.0)
    iron.set_density('g/cm3', 7.87)
    return nickel, hydrogen, iron


def room_region(padding_cm=0.0):
    """Vacuum-bounded box around the room, optionally padded outwards."""
    planes = [
        openmc.XPlane(x0=ROOM_LOWER_LEFT_CM[0] - padding_cm),
        openmc.XPlane(x0=ROOM_UPPER_RIGHT_CM[0] + padding_cm),
        openmc.YPlane(y0=ROOM_LOWER_LEFT_CM[1] - padding_cm),
        openmc.YPlane(y0=ROOM_UPPER_RIGHT_CM[1] + padding_cm),
        openmc.ZPlane(z0=ROOM_LOWER_LEFT_CM[2] - padding_cm),
        openmc.ZPlane(z0=ROOM_UPPER_RIGHT_CM[2] + padding_cm)]
    for plane in planes:
        plane.boundary_type = 'vacuum'
    xmin, xmax, ymin, ymax, zmin, zmax = planes
    return +xmin & -xmax & +ymin & -ymax & +zmin & -zmax


def configuration_at(emitter_x_cm, args, neutron_source_position_cm=None,
                     iron_slab=False):
    """Place nickel independently of any explicitly supplied neutron source.

    An emitter_x_cm of None removes the nickel from both the geometry and the
    source map. With iron_slab, a fixed activating iron slab is added.
    """
    nickel, hydrogen, iron = make_materials(
        ['nickel emitter', 'stationary hydrogen shield',
         'stationary iron slab'])
    room = room_region()
    shield_left = openmc.XPlane(x0=SHIELD_X_CM[0])
    shield_right = openmc.XPlane(x0=SHIELD_X_CM[1])
    shield_region = +shield_left & -shield_right
    void_region = room & ~shield_region
    cells = []
    components = {}
    if emitter_x_cm is None:
        # Explicit None keeps the decaying history but suppresses emission.
        components['emitter'] = None
    else:
        sphere = openmc.Sphere(x0=emitter_x_cm, r=EMITTER_RADIUS_CM)
        emitter = openmc.Cell(fill=nickel, region=-sphere)
        cells.append(emitter)
        components['emitter'] = d1s.Cells([emitter])
        void_region &= +sphere
    if iron_slab:
        # Fixed full-length slab along the bottom, 16 cm deep in z. It is
        # activated by scattered neutrons and attenuates downward photons.
        iron_bottom = openmc.YPlane(y0=IRON_Y_CM[0])
        iron_top = openmc.YPlane(y0=IRON_Y_CM[1])
        iron_front = openmc.ZPlane(z0=IRON_Z_CM[0])
        iron_back = openmc.ZPlane(z0=IRON_Z_CM[1])
        iron_region = +iron_bottom & -iron_top & +iron_front & -iron_back
        slab = openmc.Cell(fill=iron, region=room & iron_region)
        cells.append(slab)
        components['iron slab'] = d1s.Cells([slab])
        shield_region &= ~iron_region
        void_region &= ~iron_region
    shield = openmc.Cell(fill=hydrogen, region=room & shield_region)
    void = openmc.Cell(region=void_region)
    model = openmc.Model(geometry=openmc.Geometry([*cells, shield, void]))
    add_settings_and_tallies(model, args, neutron_source_position_cm)

    transforms = {}
    if emitter_x_cm is not None:
        placement = np.eye(4)
        placement[0, 3] = emitter_x_cm
        transforms['emitter'] = placement
    return d1s.Configuration(model, components=components,
                             transforms=transforms)


def add_settings_and_tallies(model, args, neutron_source_position_cm=None):
    """Apply shared run settings, the optional beam, and mesh tallies."""
    model.settings.run_mode = 'fixed source'
    model.settings.batches = args.batches
    model.settings.particles = args.particles
    model.settings.seed = 12345
    model.settings.output = {'summary': False, 'tallies': False}

    if neutron_source_position_cm is not None:
        model.settings.source = openmc.IndependentSource(
            space=openmc.stats.Point(neutron_source_position_cm),
            angle=openmc.stats.Monodirectional(NEUTRON_DIRECTION),
            energy=openmc.stats.Discrete([14.1e6], [1.0]),
            particle='neutron')

    # A fixed world-coordinate grid shared by every observation. The single
    # z bin has finite thickness, so the image is a slab-averaged flux.
    mesh = openmc.RegularMesh()
    mesh.dimension = (*args.mesh_bins, 1)
    mesh.lower_left = MESH_LOWER_LEFT_CM
    mesh.upper_right = MESH_UPPER_RIGHT_CM
    tally = openmc.Tally(name=TALLY_NAME)
    tally.filters = [openmc.MeshFilter(mesh), openmc.ParticleFilter('photon')]
    tally.scores = ['flux']
    model.tallies = [tally]
    if neutron_source_position_cm is not None:
        neutron_flux = openmc.Tally(name=NEUTRON_TALLY_NAME)
        neutron_flux.filters = [openmc.MeshFilter(mesh),
                                openmc.ParticleFilter('neutron')]
        neutron_flux.scores = ['flux']
        model.tallies.append(neutron_flux)


def mesh_flux(tally, value='mean'):
    """Return (x, y) physical flux divided by each RegularMesh voxel volume."""
    mesh = tally.find_filter(openmc.MeshFilter).mesh
    # Filter order: mesh (x, y, z), particle, nuclide, score.
    data = tally.get_reshaped_data(value=value, expand_dims=True)
    return data[:, :, 0, 0, 0, 0] / mesh.volumes[:, :, 0]


def export_animation(results, output_dir, emitter_positions, neutron_positions,
                     description, fps, irradiation_end_s, iron_slab=False):
    """Export physical mesh means/errors and a fixed-scale cooling GIF."""
    observations = list(results.observations.values())
    means, errors = [], []
    for observation in observations:
        tally = results.get_tally(TALLY_NAME, observation.step,
                                  observation.local_time)
        # D1S has already applied photons/s. Do not multiply by source rate.
        means.append(mesh_flux(tally))
        errors.append(mesh_flux(tally, 'std_dev'))
    means, errors = np.asarray(means), np.asarray(errors)
    cooling_times = np.array([o.global_time - irradiation_end_s
                             for o in observations])
    # NaN marks observations where the nickel has been removed.
    positions = np.array([np.nan if emitter_positions[o.step] is None
                          else emitter_positions[o.step]
                          for o in observations], dtype=float)
    placements = ['nickel removed' if np.isnan(x) else f'x = {x:g} cm'
                  for x in positions]
    output_dir = Path(output_dir)
    np.savez_compressed(
        output_dir / 'mesh_flux.npz', photon_flux=means,
        conditional_std_dev=errors, global_time_s=[o.global_time
                                                 for o in observations],
        time_after_shutdown_s=cooling_times, emitter_x_cm=positions,
        step_names=[o.step for o in observations],
        local_time_s=[o.local_time for o in observations],
        photon_source_rate=[o.source_rate for o in observations],
        mesh_lower_left_cm=MESH_LOWER_LEFT_CM,
        mesh_upper_right_cm=MESH_UPPER_RIGHT_CM,
        flux_units='photons/cm2/s')

    titles = [
        f'{description}\n{cooling_times[index] / 60:.1f} min after shutdown; '
        f'{placements[index]}; '
        f'Qγ = {observation.source_rate:.3e} photons/s'
        for index, observation in enumerate(observations)]
    draw_animation(means, output_dir / 'photon_flux.gif', positions,
                   neutron_positions, titles, 'Photon flux [photons/cm²/s]',
                   fps, iron_slab=iron_slab)

    print('step,minutes_after_shutdown,emitter_x_cm,'
          'photons_per_second,max_mesh_flux')
    for index, observation in enumerate(observations):
        print(f'{observation.step},{cooling_times[index] / 60:.1f},'
              f'{positions[index]:g},{observation.source_rate:.6e},'
              f'{means[index].max():.6e}')
    print(f'Animation: {output_dir / "photon_flux.gif"}')
    print(f'Mesh arrays: {output_dir / "mesh_flux.npz"}')


def export_neutron_animation(simulation, production, output_dir, fps,
                             iron_slab=False):
    """Plot the neutron flux field for each static irradiation interval."""
    means, errors, positions, sources, titles = [], [], [], [], []
    times, elapsed = [], 0.0
    for name, step in zip(simulation.step_names, simulation.steps):
        if not isinstance(step, d1s.Irradiation):
            elapsed += step.duration
            continue
        kernel = production.kernels[step.configuration]
        with openmc.StatePoint(kernel.statepoint,
                       autolink=False) as statepoint:
            tally = statepoint.get_tally(name=NEUTRON_TALLY_NAME)
            # Each example uses one unit-strength neutron source. Raw neutron
            # tallies are per initial neutron, unlike scaled D1S Results.
            means.append(mesh_flux(tally) * step.source_rate)
            errors.append(mesh_flux(tally, 'std_dev') * step.source_rate)
        positions.append(step.configuration.transforms['emitter'][0, 3])
        sources.append(step.configuration.model.settings.source[0].space.xyz)
        times.append(elapsed)
        titles.append(
            f'{name}: {elapsed / 60:g} to '
            f'{(elapsed + step.duration) / 60:g} min of irradiation\n'
            f'Source at {tuple(sources[-1])} cm; '
            f'emitter x = {positions[-1]:g} cm')
        elapsed += step.duration
    output_dir = Path(output_dir)
    np.savez_compressed(
        output_dir / 'neutron_mesh_flux.npz', neutron_flux=means,
        transport_std_dev=errors, interval_start_s=times,
        emitter_x_cm=positions, neutron_source_position_cm=sources,
        mesh_lower_left_cm=MESH_LOWER_LEFT_CM,
        mesh_upper_right_cm=MESH_UPPER_RIGHT_CM,
        flux_units='neutrons/cm2/s')
    draw_animation(np.asarray(means), output_dir / 'neutron_flux.gif',
                   positions, sources, titles, 'Neutron flux [neutrons/cm²/s]',
                   fps, neutron_active=True, iron_slab=iron_slab)
    print(f'Neutron animation: {output_dir / "neutron_flux.gif"}')


def draw_animation(means, filename, positions, neutron_positions, titles,
                   colorbar_label, fps, neutron_active=False,
                   iron_slab=False):
    """Render (frame, x, y) data with fixed axes and color normalization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle, Rectangle

    maximum = float(means.max())
    if maximum <= 0.0:
        raise RuntimeError(
            'No sampled mesh flux; increase transport histories')
    # Keep the same normalization in every frame so decay remains visible.
    norm = LogNorm(vmin=maximum * 1.e-4, vmax=maximum)
    extent = (MESH_LOWER_LEFT_CM[0], MESH_UPPER_RIGHT_CM[0],
              MESH_LOWER_LEFT_CM[1], MESH_UPPER_RIGHT_CM[1])
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    # imshow rows are y and columns are x: transpose the (x, y) mesh data.
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('black')  # Unsampled/zero bins, not missing geometry.
    image = ax.imshow(np.ma.masked_less_equal(means[0].T, 0.0),
                      origin='lower', extent=extent, norm=norm, cmap=cmap)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.axvspan(*SHIELD_X_CM, color='cyan', alpha=0.2,
               label='stationary H1 shield')
    if iron_slab:
        # Outline only, so the slab's own emission and shadow stay visible.
        ax.add_patch(Rectangle(
            (MESH_LOWER_LEFT_CM[0], IRON_Y_CM[0]),
            MESH_UPPER_RIGHT_CM[0] - MESH_LOWER_LEFT_CM[0],
            IRON_Y_CM[1] - IRON_Y_CM[0], fill=False, edgecolor='silver',
            linestyle='--', linewidth=1.5, label='stationary Fe slab'))
    positions = np.asarray(positions, dtype=float)
    emitter = Circle((0.0, 0.0), EMITTER_RADIUS_CM,
                     fill=False, edgecolor='white', linewidth=1.5,
                     label='nickel emitter')
    ax.add_patch(emitter)
    neutron_positions = np.asarray(neutron_positions)
    visible_sources = (neutron_positions[:1] if neutron_active
                       else neutron_positions)
    # Arrows start at the source point and show the beam direction.
    source_markers = ax.quiver(
        visible_sources[:, 0], visible_sources[:, 1],
        NEUTRON_DIRECTION[0] * SOURCE_ARROW_CM,
        NEUTRON_DIRECTION[1] * SOURCE_ARROW_CM, angles='xy',
        scale_units='xy', scale=1, color='lime', width=0.006,
        label='neutron beam' if neutron_active
        else 'irradiation beam sites (off)')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.legend(loc='upper right', fontsize='small')

    def update(index):
        image.set_data(np.ma.masked_less_equal(means[index].T, 0.0))
        # A NaN position means the nickel has been removed from the model.
        present = bool(np.isfinite(positions[index]))
        emitter.set_visible(present)
        if present:
            emitter.center = (positions[index], 0.0)
        if neutron_active:
            source_markers.set_offsets(neutron_positions[index:index + 1, :2])
        ax.set_title(titles[index])
        return image, emitter

    animation = FuncAnimation(fig, update, frames=len(means),
                              interval=1000 / fps, cache_frame_data=False)
    animation.save(filename, writer=PillowWriter(fps=fps))
    update(0)
    fig.savefig(filename.with_name(f'{filename.stem}_first.png'), dpi=150)
    update(len(means) - 1)
    fig.savefig(filename.with_name(f'{filename.stem}_last.png'), dpi=150)
    plt.close(fig)


def run_example(simulation, args, emitter_positions, neutron_positions,
                description, iron_slab=False):
    """Run transport before exporting its cooling mesh snapshots."""
    results = simulation.run(
        output_dir=args.output,
        run_kwargs={'threads': args.threads, 'openmc_exec': args.openmc_exec,
                    'output': False})
    irradiation_end_s = sum(step.duration for step in simulation.steps
                            if isinstance(step, d1s.Irradiation))
    export_neutron_animation(simulation, results.production, args.output,
                             args.fps, iron_slab)
    export_animation(
        results, args.output, emitter_positions, neutron_positions,
        description, args.fps, irradiation_end_s, iron_slab)
    return results