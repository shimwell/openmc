import openmc

sphere_surf_1 = openmc.Sphere(r=20)
sphere_surf_2 = openmc.Sphere(r=2, x0=10)

sphere_region_1 = -sphere_surf_1 & +sphere_surf_2  # void space
sphere_region_2 = -sphere_surf_1 & +sphere_surf_2 

sphere_cell_1 = openmc.Cell(region = sphere_region_1)

sphere_cell_2 = openmc.Cell(region = sphere_region_2)

mat_cobalt = openmc.Material()
mat_cobalt.add_element('Co', 1.)
mat_cobalt.set_density('g/cm3', 7.)
sphere_cell_2.fill = mat_cobalt

sett = openmc.Settings()
sett.batches = 10
sett.inactive = 0
sett.particles = 500
sett.run_mode = 'fixed source'

source = openmc.Source()
source.space = openmc.stats.Point((0, 0, 0))
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete([14e6], [1])
sett.source = source

r2s = openmc.R2SModel()
