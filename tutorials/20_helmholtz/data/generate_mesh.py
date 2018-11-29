# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from dolfin import *
from mshr import *
from rbnics.backends.dolfin.wrapping import counterclockwise
from rbnics.shape_parametrization.utils.symbolic import VerticesMappingIO

def generate_mesh_1():
    # Create mesh
    rectangle = Rectangle(Point(0, 0), Point(1.5, 1.))
    top_left_subdomain = Rectangle(Point(0, 0.5), Point(0.75, 1.))
    top_right_subdomain = Rectangle(Point(0.75, 0.5), Point(1.5, 1.))
    bottom_subdomain = Rectangle(Point(0.0, 0.0), Point(1.5, 0.5))
    top_left_patch = Rectangle(Point(0.325, 0.8), Point(0.425, 0.9))
    top_right_patch = Rectangle(Point(1.075, 0.8), Point(1.175, 0.9))
    
    domain = rectangle
    domain.set_subdomain(1, top_left_subdomain)
    domain.set_subdomain(2, top_right_subdomain)
    domain.set_subdomain(3, bottom_subdomain)
    domain.set_subdomain(4, top_left_patch)
    domain.set_subdomain(5, top_right_patch)
    
    mesh = generate_mesh(domain, 42)
    
    # Create subdomains
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
    
    # Create boundaries
    class OnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class Crack(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 0.75) < DOLFIN_EPS and x[1] >= 0.5
            
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    on_boundary = OnBoundary()
    on_boundary.mark(boundaries, 1)
    crack = Crack()
    crack.mark(boundaries, 1)
    
    # Save
    File("helmholtz_1.xml") << mesh
    File("helmholtz_1_physical_region.xml") << subdomains
    File("helmholtz_1_facet_region.xml") << boundaries
    XDMFFile("helmholtz_1.xdmf").write(mesh)
    XDMFFile("helmholtz_1_physical_region.xdmf").write(subdomains)
    XDMFFile("helmholtz_1_facet_region.xdmf").write(boundaries)
generate_mesh_1()

def generate_mesh_2():
    # Define domain
    rectangle = Rectangle(Point(0, 0), Point(1.5, 1.))
    domain = rectangle

    # Define vertices mappings of affine shape parametrization. These will be used
    # to partition the mesh in subdomains.
    vertices_mappings = [
        
        # Subdomains in the top left region:
        {
            ("0.325", "0.8"): ("0.325", "0.8"),
            ("0", "0.5"): ("0", "0.5"),
            ("0.425", "0.8"): ("0.425", "0.8")
        }, # subdomain 1
        {
            ("0.425", "0.8"): ("0.425", "0.8"),
            ("0", "0.5"): ("0", "0.5"),
            ("0.75", "0.5"): ("0.75", "mu[1]")
        }, # subdomain 2
        {
            ("0.325", "0.8"): ("0.325", "0.8"),
            ("0.325", "0.9"): ("0.325", "0.9"),
            ("0", "0.5"): ("0", "0.5")
        }, # subdomain 3
        {
            ("0.325", "0.9"): ("0.325", "0.9"),
            ("0", "1"): ("0", "1"),
            ("0", "0.5"): ("0", "0.5")
        }, # subdomain 4
        {
            ("0.425", "0.8"): ("0.425", "0.8"),
            ("0.75", "0.5"): ("0.75", "mu[1]"),
            ("0.425", "0.9"): ("0.425", "0.9")
        }, # subdomain 5
        {
            ("0.75", "1"): ("0.75", "1"),
            ("0.425", "0.9"): ("0.425", "0.9"),
            ("0.75", "0.5"): ("0.75", "mu[1]")
        }, # subdomain 6
        {
            ("0.325", "0.9"): ("0.325", "0.9"),
            ("0.425", "0.9"): ("0.425", "0.9"),
            ("0", "1"): ("0", "1")
        }, # subdomain 7
        {
            ("0.75", "1"): ("0.75", "1"),
            ("0", "1"): ("0", "1"),
            ("0.425", "0.9"): ("0.425", "0.9")
        }, # subdomain 8
        
        # Subdomains in the top right region:
        {
            ("1.075", "0.8"): ("1.075", "0.8"),
            ("0.75", "0.5"): ("0.75", "mu[1]"),
            ("1.175", "0.8"): ("1.175", "0.8")
        }, # subdomain 9
        {
            ("1.175", "0.8"): ("1.175", "0.8"),
            ("0.75", "0.5"): ("0.75", "mu[1]"),
            ("1.5", "0.5"): ("1.5", "0.5")
        }, # subdomain 10
        {
            ("1.075", "0.8"): ("1.075", "0.8"),
            ("1.075", "0.9"): ("1.075", "0.9"),
            ("0.75", "0.5"): ("0.75", "mu[1]")
        }, # subdomain 11
        {
            ("1.075", "0.9"): ("1.075", "0.9"),
            ("0.75", "1"): ("0.75", "1"),
            ("0.75", "0.5"): ("0.75", "mu[1]")
        }, # subdomain 12
        {
            ("1.175", "0.8"): ("1.175", "0.8"),
            ("1.5", "0.5"): ("1.5", "0.5"),
            ("1.175", "0.9"): ("1.175", "0.9")
        }, # subdomain 13
        {
            ("1.5", "1"): ("1.5", "1"),
            ("1.175", "0.9"): ("1.175", "0.9"),
            ("1.5", "0.5"): ("1.5", "0.5")
        }, # subdomain 14
        {
            ("1.075", "0.9"): ("1.075", "0.9"),
            ("1.175", "0.9"): ("1.175", "0.9"),
            ("0.75", "1"): ("0.75", "1")
        }, # subdomain 15
        {
            ("1.5", "1"): ("1.5", "1"),
            ("0.75", "1"): ("0.75", "1"),
            ("1.175", "0.9"): ("1.175", "0.9")
        }, # subdomain 16
        
        # Subdomains in the bottom region:
        {
            ("0", "0"): ("0.", "0"),
            ("0.75", "0.5"): ("0.75", "mu[1]"),
            ("0", "0.5"): ("0", "0.5")
        }, # subdomain 17
        {
            ("0", "0"): ("0", "0"),
            ("1.5", "0"): ("1.5", "0"),
            ("0.75", "0.5"): ("0.75", "mu[1]"),
        }, # subdomain 18
        {
            ("0.75", "0.5"): ("0.75", "mu[1]"),
            ("1.5", "0"): ("1.5", "0"),
            ("1.5", "0.5"): ("1.5", "0.5"),
        }, # subdomain 19
        
        # Subdomains in the top left patch:
        {
            ("0.325", "0.8"): ("0.325", "0.8"),
            ("0.325", "0.9"): ("0.325", "0.9"),
            ("0.425", "0.8"): ("0.425", "0.8"),
        }, # subdomain 20
        {
            ("0.425", "0.9"): ("0.425", "0.9"),
            ("0.325", "0.9"): ("0.325", "0.9"),
            ("0.425", "0.8"): ("0.425", "0.8"),
        }, # subdomain 21
        
        # Subdomains in the top right patch:
        {
            ("1.075", "0.8"): ("1.075", "0.8"),
            ("1.075", "0.9"): ("1.075", "0.9"),
            ("1.175", "0.8"): ("1.175", "0.8"),

        }, # subdomain 22
        {
            ("1.075", "0.9"): ("1.075", "0.9"),
            ("1.175", "0.8"): ("1.175", "0.8"),
            ("1.175", "0.9"): ("1.175", "0.9"),

        } # subdomain 23
    ]
    
    # Create mesh
    for i, vertices_mapping in enumerate(vertices_mappings):
        subdomain_i = Polygon([Point(*[float(coord) for coord in vertex]) for vertex in counterclockwise(vertices_mapping.keys())])
        domain.set_subdomain(i + 1, subdomain_i)
    mesh = generate_mesh(domain, 42)

    # Create subdomains
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), mesh.domains())
    
    # Create boundaries
    class OnBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class Crack(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 0.75) < DOLFIN_EPS and x[1] >= 0.5
            
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    on_boundary = OnBoundary()
    on_boundary.mark(boundaries, 1)
    crack = Crack()
    crack.mark(boundaries, 1)

    # Save
    VerticesMappingIO.save_file(vertices_mappings, ".", "helmholtz_2_vertices_mapping.vmp")
    File("helmholtz_2.xml") << mesh
    File("helmholtz_2_physical_region.xml") << subdomains
    File("helmholtz_2_facet_region.xml") << boundaries
    XDMFFile("helmholtz_2.xdmf").write(mesh)
    XDMFFile("helmholtz_2_physical_region.xdmf").write(subdomains)
    XDMFFile("helmholtz_2_facet_region.xdmf").write(boundaries)
generate_mesh_2()
