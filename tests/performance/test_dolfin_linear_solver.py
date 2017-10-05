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

import pytest
from numpy import isclose
from dolfin import assemble, dx, Function, FunctionSpace, grad, inner, solve, TestFunction, TrialFunction, UnitSquareMesh
from rbnics.backends import LinearSolver as FactoryLinearSolver
from rbnics.backends.dolfin import LinearSolver as DolfinLinearSolver
from test_utils import RandomDolfinFunction

LinearSolver = None
AllLinearSolver = {"dolfin": FactoryLinearSolver, "factory": DolfinLinearSolver}

class Data(object):
    def __init__(self, Th, callback_type):
        # Create mesh and define function space
        mesh = UnitSquareMesh(Th, Th)
        self.V = FunctionSpace(mesh, "Lagrange", 1)
        # Define variational problem
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.a = inner(grad(u), grad(v))*dx + inner(u, v)*dx
        self.f = lambda g: g*v*dx
        # Define callback function depending on callback type
        assert callback_type in ("form callbacks", "tensor callbacks")
        if callback_type == "form callbacks":
            def callback(arg):
                return arg
        elif callback_type == "tensor callbacks":
            def callback(arg):
                return assemble(arg)
        self.callback_type = callback_type
        self.callback = callback
        
    def generate_random(self):
        # Generate random rhs
        g = RandomDolfinFunction(self.V)
        # Return
        return (self.callback(self.a), self.callback(self.f(g)))
        
    def evaluate_builtin(self, a, f):
        result_builtin = Function(self.V)
        if self.callback_type == "form callbacks":
            solve(a == f, result_builtin, solver_parameters={"linear_solver": "mumps"})
        elif self.callback_type == "tensor callbacks":
            solve(a, result_builtin.vector(), f, "mumps")
        return result_builtin
        
    def evaluate_backend(self, a, f):
        result_backend = Function(self.V)
        solver = LinearSolver(a, result_backend, f)
        solver.solve()
        return result_backend
        
    def assert_backend(self, a, f, result_backend):
        result_builtin = self.evaluate_builtin(a, f)
        error = Function(self.V)
        error.vector().add_local(+ result_backend.vector().array())
        error.vector().add_local(- result_builtin.vector().array())
        error.vector().apply("add")
        relative_error = error.vector().norm("l2")/result_builtin.vector().norm("l2")
        assert isclose(relative_error, 0., atol=1e-12)
        
@pytest.mark.parametrize("Th", [2**i for i in range(3, 10)])
@pytest.mark.parametrize("callback_type", ["form callbacks", "tensor callbacks"])
@pytest.mark.parametrize("test_type", ["builtin"] + list(AllLinearSolver.keys()))
def test_dolfin_linear_solver(Th, callback_type, test_type, benchmark):
    data = Data(Th, callback_type)
    print("Th = " + str(Th) + ", Nh = " + str(data.V.dim()))
    if test_type == "builtin":
        print("Testing " + test_type + ", callback_type = " + callback_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing " + test_type + " backend" + ", callback_type = " + callback_type)
        global LinearSolver
        LinearSolver = AllLinearSolver[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
