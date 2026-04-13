import openmc

openmc.config['cross_sections'] = 'cross_sections.xml'

mat1 = openmc.Material(name='mat1')
mat1.add_nuclide('Li7', 0.95)
mat1.set_density('g/cm3', 1.0)

materials = openmc.Materials([mat1])

bound_dag_univ = openmc.DAGMCUniverse(filename='small_dagmc_file.h5m').bounded_universe()
my_geometry = openmc.Geometry(root=bound_dag_univ)

umesh = openmc.UnstructuredMesh(filename='small_um.vtk', library='moab')
umesh.id = 1

mesh_filter = openmc.MeshFilter(umesh)

# Create flux mesh tally to score alpha production
mesh_tally = openmc.Tally(tally_id=1, name='alpha_production_on_mesh')  # note the tally_id is specified
mesh_tally.filters = [mesh_filter]
mesh_tally.scores = ['(n,Xa)']  # where X is a wild card


tallies = openmc.Tallies([mesh_tally])


my_source = openmc.IndependentSource()
my_source.space = openmc.stats.Point((0.4, 0, 0.4))


settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.batches = 3
settings.particles = 1000
settings.source = my_source
settings.photon_transport=True

my_model = openmc.Model(
    materials=materials,
    geometry=my_geometry,
    settings=settings,
    tallies=tallies
)

statepoint_file = my_model.run()

statepoint = openmc.StatePoint(statepoint_file)

my_tally = statepoint.get_tally(name='alpha_production_on_mesh')

umesh_from_sp = statepoint.meshes[1] # note to self we can add a function to openmc to get the mesh by type or name

# needed to trigger internal mesh data loading with openmc v0.15, fixed on dev
# note to self we can add a function to openmc to get this automated when writting to vtk
centroids = umesh_from_sp.centroids
mesh_vols = umesh_from_sp.volumes

umesh_from_sp.write_data_to_vtk(
    datasets={'mean': my_tally.mean.flatten()},
    filename = "shape_alpha_production_on_mesh.vtkhdf",
)
# vtk and vtk are not used as package does not have optional vtk dependency
# umesh_from_sp.write_data_to_vtk(
#     datasets={'mean': my_tally.mean.flatten()},
#     filename = "shape_alpha_production_on_mesh.vtk",
#     filename = "shape_alpha_production_on_mesh.vtu",
# )