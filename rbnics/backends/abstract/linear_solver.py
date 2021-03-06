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

from rbnics.utils.decorators import ABCMeta, AbstractBackend, abstractmethod

@AbstractBackend
class LinearSolver(object, metaclass=ABCMeta):
    # will use @overload in derived classes
    def __init__(self, lhs, solution, rhs, bcs=None):
        pass
        
    # will use @overload in derived classes
    def __init__(self, problem_wrapper, solution):
        pass
        
    @abstractmethod
    def set_parameters(self, parameters):
        pass
        
    @abstractmethod
    def solve(self):
        pass
        
class LinearProblemWrapper(object, metaclass=ABCMeta):
    @abstractmethod
    def matrix_eval(self):
        pass
        
    @abstractmethod
    def vector_eval(self):
        pass
    
    @abstractmethod
    def bc_eval(self):
        pass
        
    @abstractmethod
    def monitor(self, solution):
        pass
