"""Example 1: fixed geometry irradiated at three neutron source positions."""

import numpy as np
from openmc.deplete import d1s

from d1s_mesh_common import configuration_at, parse_arguments, run_example


NEUTRON_POSITIONS_CM = ((-12.0, -4.0, 0.0),
                        (-12.0, 0.0, 0.0),
                        (-12.0, 4.0, 0.0))


def make_simulation(args):
    # Same physical geometry, materials, and mesh in every neutron model.
    # Changing source position requires a distinct production configuration.
    irradiations = [
        d1s.Irradiation(
            configuration_at(emitter_x_cm=0.0, args=args,
                             neutron_source_position_cm=source_position),
            (20.0, 'min'), source_rate=1.e12, name=f'irradiation-{index}')
        for index, source_position in enumerate(NEUTRON_POSITIONS_CM, 1)
    ]
    shutdown = configuration_at(emitter_x_cm=0.0, args=args)
    simulation = d1s.Simulation([
        *irradiations,
        d1s.Cooling(shutdown, (1.0, 'h'),
                    sample_times=np.linspace(0.0, 3600.0, 12),
                    name='stationary-emitter'),
    ], chain_file=args.chain_file)
    return simulation, {'stationary-emitter': 0.0}


def main():
    args = parse_arguments(__doc__, 'd1s-moving-neutron-source')
    simulation, emitter_positions = make_simulation(args)
    run_example(simulation, args, emitter_positions, NEUTRON_POSITIONS_CM,
                'Fixed geometry; earlier neutron source movement')


if __name__ == '__main__':
    main()