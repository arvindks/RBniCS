# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import Mesh, Point
from RBniCS.backends.abstract import ReducedVertices as AbstractReducedVertices
from RBniCS.utils.decorators import BackendFor, Extends, override
from RBniCS.utils.io import ExportableList

@Extends(AbstractReducedVertices)
@BackendFor("fenics", inputs=(Mesh, ))
class ReducedVertices(AbstractReducedVertices):
    def __init__(self, mesh, original_vertex_list=None, original_component_list=None):
        AbstractReducedVertices.__init__(self, mesh)
        self._mesh = mesh
        # Vertex storage
        if original_vertex_list is None:
            self._vertex_list = ExportableList("pickle") # list of vertices
            self._component_list = ExportableList("pickle") # list of function components
        else:
            self._vertex_list = original_vertex_list
            self._component_list = original_component_list
        # Additional storage to detect local vertices
        self._bounding_box_tree = mesh.bounding_box_tree()
        self._mpi_comm = mesh.mpi_comm().tompi4py()
        self._is_local = dict()
        
    @override
    def append(self, vertex_and_component):
        assert isinstance(vertex_and_component, tuple)
        assert len(vertex_and_component) == 2
        vertex = vertex_and_component[0]
        component = vertex_and_component[1]
        self._vertex_list.append(vertex)
        self._component_list.append(component)
        
    @override
    def save(self, directory, filename):
        self._vertex_list.save(directory, filename + "_vertices")
        self._component_list.save(directory, filename + "_components")
        
    @override
    def load(self, directory, filename):
        vertex_import_successful = self._vertex_list.load(directory, filename + "_vertices")
        component_import_successful = self._component_list.load(directory, filename + "_components")
        return vertex_import_successful and component_import_successful
        
    @override
    def __getitem__(self, key):
        assert isinstance(key, slice)
        assert key.start is None 
        assert key.step is None
        return ReducedVertices(self._mesh, self._vertex_list[key], self._component_list[key])
                
    def is_local(self, index):
        try:
            return self._is_local[index]
        except KeyError:
            self._is_local[index] = self._bounding_box_tree.collides_entity(Point(self._vertex_list[index]))
            return self._is_local[index]
        
