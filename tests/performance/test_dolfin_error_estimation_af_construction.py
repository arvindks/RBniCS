# Copyright (C) 2015-2017 by the RBniCS authors
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

from __future__ import print_function
from dolfin import *
from numpy.linalg import norm
from rbnics.backends import BasisFunctionsMatrix, transpose
from rbnics.backends.online.numpy import Vector as NumpyVector
from test_utils import RandomDolfinFunction, TestBase

class Test(TestBase):
    def __init__(self, Nh, N):
        self.N = N
        mesh = UnitSquareMesh(Nh, Nh)
        self.V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.a = lambda k: k*inner(grad(u), grad(v))*dx
        # Call parent init
        TestBase.__init__(self)
            
    def run(self):
        N = self.N
        test_id = self.test_id
        test_subid = self.test_subid
        if test_id >= 0:
            if not self.index in self.storage:
                # Generate random vectors
                Z = BasisFunctionsMatrix(self.V)
                Z.init("u")
                for i in range(self.N):
                    b = RandomDolfinFunction(self.V)
                    Z.enrich(b)
                # Generate random matrix
                k = RandomDolfinFunction(self.V)
                A = assemble(self.a(k))
                # Store
                z = RandomDolfinFunction(self.V)
                self.storage[self.index] = (Z, A, z.vector())
            else:
                (Z, A, z) = self.storage[self.index]
            self.index += 1
        if test_id >= 1:
            if test_id > 1 or (test_id == 1 and test_subid == "a"):
                # Time using built in methods
                Z_T_dot_A_z_builtin = NumpyVector(self.N)
                A_z = A*z
                for i in range(self.N):
                    Z_T_dot_A_z_builtin[i] = Z[i].vector().inner(A_z)
            if test_id > 1 or (test_id == 1 and test_subid == "b"):
                # Time using transpose() method
                Z_T_dot_A_z_transpose = transpose(Z)*A*z
        if test_id >= 2:
            return norm(Z_T_dot_A_z_builtin - Z_T_dot_A_z_transpose)/norm(Z_T_dot_A_z_builtin)

for i in range(3, 7):
    Nh = 2**i
    for j in range(1, 4):
        N = 10 + 4*j
        test = Test(Nh, N)
        print("Nh =", test.V.dim(), "and N =", N)
        
        test.init_test(0)
        (usec_0_build, usec_0_access) = test.timeit()
        print("Construction:", usec_0_build, "usec", "(number of runs: ", test.number_of_runs(), ")")
        print("Access:", usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        test.init_test(1, "a")
        usec_1a = test.timeit()
        print("Builtin method:", usec_1a - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        test.init_test(1, "b")
        usec_1b = test.timeit()
        print("transpose() method:", usec_1b - usec_0_access, "usec", "(number of runs: ", test.number_of_runs(), ")")
        
        print("Relative overhead of the transpose() method:", (usec_1b - usec_1a)/(usec_1a - usec_0_access))
        
        test.init_test(2)
        error = test.average()
        print("Relative error:", error)