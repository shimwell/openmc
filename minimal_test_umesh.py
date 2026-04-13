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

mesh_tally = openmc.Tally(tally_id=1, name='alpha_production_on_mesh')
mesh_tally.filters = [mesh_filter]
mesh_tally.scores = ['(n,Xa)']

tallies = openmc.Tallies([mesh_tally])

my_source = openmc.IndependentSource()
my_source.space = openmc.stats.Point((0.4, 0, 0.4))

settings = openmc.Settings()
settings.run_mode = 'fixed source'
settings.batches = 3
settings.particles = 1000
settings.source = my_source
settings.photon_transport = True

my_model = openmc.Model(
    materials=materials,
    geometry=my_geometry,
    settings=settings,
    tallies=tallies
)

statepoint_file = my_model.run()

statepoint = openmc.StatePoint(statepoint_file)
my_tally = statepoint.get_tally(name='alpha_production_on_mesh')

print("=== UnstructuredMesh (moab) ===")
for key, value in statepoint.runtime.items():
    print(f"  {key}: {value:.4f}s")
print(f"Tally mean sum: {my_tally.mean.sum():.6e}")
