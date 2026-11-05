"""Example 3 in CAD: the same schedule with DAGMC geometry from CadQuery."""

from functools import partial
from pathlib import Path

import numpy as np
import openmc
from openmc.deplete import d1s

from d1s_mesh_common import (
    EMITTER_RADIUS_CM, IRON_Y_CM, IRON_Z_CM, ROOM_LOWER_LEFT_CM,
    ROOM_UPPER_RIGHT_CM, SHIELD_X_CM, add_settings_and_tallies,
    make_materials, parse_arguments, room_region, run_example)
from moving_then_removed_volume import NEUTRON_POSITION_CM, make_simulation


# Raw H5M volume IDs follow the order in which solids are added.
IRON_VOLUME_ID = 1
NICKEL_VOLUME_ID = 3
MATERIAL_TAGS = ('nickel', 'hydrogen', 'iron')
# The CSG box around the DAGMC universe is padded so that no CAD surface
# coincides with the vacuum boundary. The extra space is void.
ROOM_PADDING_CM = 1.0


def _box(lower_left, upper_right):
    import cadquery as cq
    lower_left, upper_right = np.asarray(lower_left), np.asarray(upper_right)
    size = upper_right - lower_left
    return (cq.Workplane().box(*size, centered=False)
            .translate(tuple(lower_left)))


def export_h5m(emitter_x_cm, cad_dir):
    """Mesh the CadQuery solids for one nickel placement into a DAGMC file."""
    import cadquery as cq
    from cad_to_dagmc import CadToDagmc

    name = 'removed' if emitter_x_cm is None else f'x{emitter_x_cm:g}'
    filename = Path(cad_dir) / f'nickel-{name}.h5m'
    if filename.exists():
        return filename
    iron = _box((ROOM_LOWER_LEFT_CM[0], IRON_Y_CM[0], IRON_Z_CM[0]),
                (ROOM_UPPER_RIGHT_CM[0], IRON_Y_CM[1], IRON_Z_CM[1]))
    # The H1 slab spans the room height and depth, minus the iron it meets.
    hydrogen = _box(
        (SHIELD_X_CM[0], ROOM_LOWER_LEFT_CM[1], ROOM_LOWER_LEFT_CM[2]),
        (SHIELD_X_CM[1], ROOM_UPPER_RIGHT_CM[1], ROOM_UPPER_RIGHT_CM[2])
    ).cut(iron)
    cad = CadToDagmc()
    cad.add_cadquery_object(iron.val(), material_tags=['iron'])
    cad.add_cadquery_object(hydrogen.val(), material_tags=['hydrogen'])
    if emitter_x_cm is not None:
        nickel = cq.Workplane().sphere(EMITTER_RADIUS_CM).translate(
            (emitter_x_cm, 0.0, 0.0))
        cad.add_cadquery_object(nickel.val(), material_tags=['nickel'])
    filename.parent.mkdir(parents=True, exist_ok=True)
    cad.export_dagmc_h5m_file(
        filename=str(filename), meshing_backend='cad-to-dagmc-mesher',
        tolerance=0.01, angular_tolerance=0.1)
    return filename


def cad_configuration_at(emitter_x_cm, args, neutron_source_position_cm=None,
                         iron_slab=True, cad_dir=None):
    """DAGMC version of configuration_at; the iron slab is always present."""
    if not iron_slab:
        raise ValueError('The CAD model always contains the iron slab')
    filename = export_h5m(emitter_x_cm, cad_dir)
    materials = make_materials(MATERIAL_TAGS)
    # auto_geom_ids avoids clashes with CSG cell IDs. D1S selectors use the
    # raw H5M volume IDs, which are unaffected.
    universe = openmc.DAGMCUniverse(filename, auto_geom_ids=True)
    room = openmc.Cell(region=room_region(ROOM_PADDING_CM), fill=universe)
    used = [m for m in materials if m.name in universe.material_names]
    model = openmc.Model(geometry=openmc.Geometry([room]),
                         materials=openmc.Materials(used))
    add_settings_and_tallies(model, args, neutron_source_position_cm)

    components = {'iron slab': d1s.DAGMCVolumes(universe, [IRON_VOLUME_ID])}
    transforms = {}
    if emitter_x_cm is None:
        # Explicit None keeps the decaying history but suppresses emission.
        components['emitter'] = None
    else:
        components['emitter'] = d1s.DAGMCVolumes(universe,
                                                 [NICKEL_VOLUME_ID])
        placement = np.eye(4)
        placement[0, 3] = emitter_x_cm
        transforms['emitter'] = placement
    # Keep the same key order as the CSG example.
    components = {key: components[key] for key in ('emitter', 'iron slab')}
    return d1s.Configuration(model, components=components,
                             transforms=transforms)


def main():
    args = parse_arguments(__doc__, 'd1s-moving-then-removed-volume-cad',
                           cad=True)
    configuration = partial(cad_configuration_at, cad_dir=args.cad_dir)
    simulation, emitter_positions = make_simulation(args, configuration)
    run_example(simulation, args, emitter_positions, [NEUTRON_POSITION_CM],
                'CAD (DAGMC): fixed Fe slab; nickel moves then is removed',
                iron_slab=True)


if __name__ == '__main__':
    main()
