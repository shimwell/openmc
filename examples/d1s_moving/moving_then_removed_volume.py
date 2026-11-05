"""Example 3: activated nickel moves then is removed above a fixed iron slab."""

from openmc.deplete import d1s

from d1s_mesh_common import configuration_at, parse_arguments, run_example


# This source is independent of the nickel position and never moves.
NEUTRON_POSITION_CM = (-15.0, 0.0, 0.0)
IRRADIATION_POSITIONS_CM = (0.0, 12.0, 24.0)
# Nickel positions during cooling. None removes it from the model.
COOLING_POSITIONS_CM = (24.0, 16.0, 8.0, None)
REMOVED_SAMPLE_MINUTES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0)


def make_simulation(args, configuration_at=configuration_at):
    """Build the schedule. configuration_at may build CSG or CAD models."""
    irradiations = [
        d1s.Irradiation(
            configuration_at(emitter_x_cm=emitter_position, args=args,
                             neutron_source_position_cm=NEUTRON_POSITION_CM,
                             iron_slab=True),
            (20.0, 'min'), source_rate=1.e12, name=f'irradiation-{index}')
        for index, emitter_position in enumerate(IRRADIATION_POSITIONS_CM, 1)
    ]
    # The 'emitter' and 'iron slab' identities are present in every model.
    # The iron never moves, so it needs no transform.
    cooling = []
    emitter_positions = {}
    for index, emitter_position in enumerate(COOLING_POSITIONS_CM, 1):
        configuration = configuration_at(emitter_x_cm=emitter_position,
                                         args=args, iron_slab=True)
        if emitter_position is None:
            # Nickel is gone from both geometry and source map. Only the
            # activated iron slab still emits decay photons.
            name = 'nickel-removed'
            cooling.append(d1s.Cooling(
                configuration, (40.0, 'min'), name=name,
                sample_times=[(t, 'min') for t in REMOVED_SAMPLE_MINUTES]))
        else:
            name = f'cooling-position-{index}'
            cooling.append(d1s.Cooling(
                configuration, (10.0, 'min'), name=name,
                sample_times=[0.0, (5.0, 'min')]))
        emitter_positions[name] = emitter_position
    simulation = d1s.Simulation([*irradiations, *cooling],
                                 chain_file=args.chain_file)
    return simulation, emitter_positions


def main():
    args = parse_arguments(__doc__, 'd1s-moving-then-removed-volume')
    simulation, emitter_positions = make_simulation(args)
    run_example(simulation, args, emitter_positions, [NEUTRON_POSITION_CM],
                'Fixed Fe slab; nickel moves then is removed', iron_slab=True)


if __name__ == '__main__':
    main()
