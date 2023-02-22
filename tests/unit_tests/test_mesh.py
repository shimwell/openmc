import openmc
import pytest
import numpy as np

@pytest.mark.parametrize("val_left,val_right", [(0, 0), (-1., -1.), (2.0, 2)])
def test_raises_error_when_flat(val_left, val_right):
    """Checks that an error is raised when a mesh is flat"""
    mesh = openmc.RegularMesh()

    # Same X
    with pytest.raises(ValueError):
        mesh.lower_left = [val_left, -25, -25]
        mesh.upper_right = [val_right, 25, 25]

    with pytest.raises(ValueError):
        mesh.upper_right = [val_right, 25, 25]
        mesh.lower_left = [val_left, -25, -25]

    # Same Y
    with pytest.raises(ValueError):
        mesh.lower_left = [-25, val_left, -25]
        mesh.upper_right = [25, val_right, 25]

    with pytest.raises(ValueError):
        mesh.upper_right = [25, val_right, 25]
        mesh.lower_left = [-25, val_left, -25]

    # Same Z
    with pytest.raises(ValueError):
        mesh.lower_left = [-25, -25, val_left]
        mesh.upper_right = [25, 25, val_right]

    with pytest.raises(ValueError):
        mesh.upper_right = [25, 25, val_right]
        mesh.lower_left = [-25, -25, val_left]


def test_reg_mesh_from_slice_of_domain():
    """Tests a RegularMesh can be made from a Cell and the specified dimensions
    slice_value and slice_width are propagated through correctly."""
    surface = openmc.Sphere(r=10)
    cell = openmc.Cell(region=-surface)

    mesh = openmc.RegularMesh.from_slice_of_domain(
        domain=cell,
        axis_dimension={'x': 3, 'y': 9},
        slice_value=5,
        slice_width=3,
    )
    assert isinstance(mesh, openmc.RegularMesh)
    assert np.array_equal(mesh.dimension, (3, 9, 1))
    assert np.array_equal(mesh.lower_left, [-10, -10, 5-3/2])
    assert np.array_equal(mesh.upper_right, [10, 10, 5+3/2])


def test_reg_mesh_from_slice_of_domain_errors():
    """Tests a RegularMesh raises ValueErrors for incorrect inputs."""
    surface = openmc.Sphere(r=10)
    cell = openmc.Cell(region=-surface)

    # two keys that are the same
    with pytest.raises(ValueError):
        openmc.RegularMesh.from_slice_of_domain(
            domain=cell,
            axis_dimension={'y': 3, 'y': 9},
        )

    # incorrect key
    with pytest.raises(ValueError):
        openmc.RegularMesh.from_slice_of_domain(
            domain=cell,
            axis_dimension={'x-axis': 3, 'y': 9},
        )

    # three keys
    with pytest.raises(ValueError):
        openmc.RegularMesh.from_slice_of_domain(
            domain=cell,
            axis_dimension={'x': 3, 'y': 9, 'z': 1},
        )

    # axis_dimension specified as a dimension
    with pytest.raises(TypeError):
        openmc.RegularMesh.from_slice_of_domain(
            domain=cell,
            axis_dimension=[3, 9],
        )
