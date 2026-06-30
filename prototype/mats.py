"""Shared 10-material test matrix for transport-free MGXS validation.
Spans heavy shield, structural, breeder, coolant and hydrogenous materials."""
import openmc

def materials():
    out = []
    w = openmc.Material(name="tungsten"); w.add_element("W", 1.0); w.set_density("g/cm3", 19.3); out.append(w)
    s = openmc.Material(name="steel"); s.add_element("Fe",.70); s.add_element("Cr",.18); s.add_element("Ni",.12); s.set_density("g/cm3",7.9); out.append(s)
    f = openmc.Material(name="Fe56"); f.add_nuclide("Fe56",1.0); f.set_density("g/cm3",7.87); out.append(f)
    z = openmc.Material(name="CuCrZr"); z.add_element("Cu",0.9905,percent_type="wo"); z.add_element("Cr",0.008,percent_type="wo"); z.add_element("Zr",0.0015,percent_type="wo"); z.set_density("g/cm3",8.9); out.append(z)
    zr = openmc.Material(name="Zircaloy"); zr.add_element("Zr",0.9823,percent_type="wo"); zr.add_element("Sn",0.0145,percent_type="wo"); zr.add_element("Fe",0.0021,percent_type="wo"); zr.add_element("Cr",0.001,percent_type="wo"); zr.set_density("g/cm3",6.56); out.append(zr)
    sic = openmc.Material(name="SiC"); sic.add_elements_from_formula("SiC"); sic.set_density("g/cm3",3.21); out.append(sic)
    lso = openmc.Material(name="Li4SiO4"); lso.add_elements_from_formula("Li4SiO4"); lso.set_density("g/cm3",2.40); out.append(lso)  # fusion breeder ceramic
    cc = openmc.Material(name="concrete")
    for el, wo in [("H",.01),("O",.529),("Si",.337),("Ca",.044),("Al",.034),("Fe",.014),("Na",.016),("K",.013),("Mg",.002),("C",.001)]:
        cc.add_element(el, wo, percent_type="wo")
    cc.set_density("g/cm3", 2.3); out.append(cc)
    h2o = openmc.Material(name="H2O"); h2o.add_elements_from_formula("H2O"); h2o.set_density("g/cm3",1.0); out.append(h2o)
    he = openmc.Material(name="Helium"); he.add_element("He",1.0); he.set_density("g/cm3",0.00311); out.append(he)  # 5 MPa, 500 C ideal-gas
    return out

NAMES = [m.name for m in materials()]
