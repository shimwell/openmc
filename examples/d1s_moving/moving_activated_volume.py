"""Example 2: fixed neutron source, one moving nickel activation volume."""

from openmc.deplete import d1s

from d1s_mesh_common import configuration_at, parse_arguments, run_example


# This source is independent of the nickel position and never moves.
NEUTRON_POSITION_CM = (-15.0, 0.0, 0.0)
IRRADIATION_POSITIONS_CM = (0.0, 12.0, 24.0)
COOLING_POSITIONS_CM = (24.0, 18.0, 12.0, 6.0, 0.0, -6.0)


def make_simulation(args):
    irradiations = [
        d1s.Irradiation(
            configuration_at(emitter_x_cm=emitter_position, args=args,
                             neutron_source_position_cm=NEUTRON_POSITION_CM),
            (20.0, 'min'), source_rate=1.e12, name=f'irradiation-{index}')
        for index, emitter_position in enumerate(IRRADIATION_POSITIONS_CM, 1)
    ]
    # Stable 'emitter' identity follows the same nickel object through all
    # irradiation and cooling models. History does not reset after a move.
    cooling = []
    emitter_positions = {}
    for index, emitter_position in enumerate(COOLING_POSITIONS_CM, 1):
        name = f'cooling-position-{index}'
        emitter_positions[name] = emitter_position
        cooling.append(d1s.Cooling(
            configuration_at(emitter_x_cm=emitter_position, args=args),
            (10.0, 'min'), sample_times=[0.0, (5.0, 'min')], name=name))
    simulation = d1s.Simulation([*irradiations, *cooling],
                                 chain_file=args.chain_file)
    return simulation, emitter_positions


def main():
    args = parse_arguments(__doc__, 'd1s-moving-activated-volume')
    simulation, emitter_positions = make_simulation(args)
    run_example(simulation, args, emitter_positions, [NEUTRON_POSITION_CM],
                'Fixed neutron source; moving activated nickel volume')


if __name__ == '__main__':
    main()