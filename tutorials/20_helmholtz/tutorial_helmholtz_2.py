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
from rbnics import *

@ExactStabilityFactor()
@PullBackFormsToReferenceDomain()
@AffineShapeParametrization("data/helm_vertices_mapping.vmp")
class Helmholtz(EllipticProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        EllipticCoerciveCompliantProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        self.subdomains = subdomains
        self.boundaries = boundaries
        """
        self._eigen_solver_parameters.update({
            "bounding_box_minimum": {"problem_type": "gen_hermitian", "spectral_transform": "shift-and-invert", "spectral_shift": 1.e-5},
            "bounding_box_maximum": {"problem_type": "gen_hermitian", "spectral_transform": "shift-and-invert", "spectral_shift": 1.e5},
            "stability_factor": {"problem_type": "gen_hermitian", "spectral_transform": "shift-and-invert", "spectral_shift": 1.e-5}
        })
        """ # TODO uncomment

    def get_stability_factor_lower_bound(self):
        return 1. # TODO remove
        
    def name(self):
        return "Helmholtz2"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        mu = self.mu
        if term == "a":
            theta_a0 = 1.
            theta_a1 = - mu[0]**2
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = 1.
            return (theta_f0,)
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "a":
            u = self.u
            a0 = inner(grad(u), grad(v))*dx
            a1 = inner(u, v)*dx
            return (a0, a1)
        elif term == "f":
            f0 = v*dx(20) + v*dx(21)
            return (f0,)
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, Constant(0.0), self.boundaries, 1)]
            return (bc0,)
        elif term == "inner_product":
            u = self.u
            x0 = inner(u, v)*dx + inner(grad(u), grad(v))*dx
            return (x0,)
        else:
            raise ValueError("Invalid term for assemble_operator().")

# 1. Read the mesh for this problem
mesh = Mesh("data/helmholtz_2.xml")
subdomains = MeshFunction("size_t", mesh, "data/helmholtz_2_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/helmholtz_2_facet_region.xml")

# 2. Create Finite Element space
V = FunctionSpace(mesh, "Lagrange", 2)

# 3. Allocate an object of the Helmholtz_param class
helmholtz_problem = Helmholtz(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(1., 20.), (0.2, 0.8)]
helmholtz_problem.set_mu_range(mu_range)

# 4. Prepare reduction with a reduced basis method
reduced_basis_method = ReducedBasis(helmholtz_problem)
reduced_basis_method.set_Nmax(50)
reduced_basis_method.set_tolerance(1e-4)

# 5. Perform the offline phase
reduced_basis_method.initialize_training_set(100)
reduced_helmholtz_problem = reduced_basis_method.offline()

# 6. Perform an online solve
online_mu = (10., 0.5)
reduced_helmholtz_problem.set_mu(online_mu)
reduced_helmholtz_problem.solve()
reduced_helmholtz_problem.export_solution(filename="online_solution")

# 7. Perform an error analysis
reduced_basis_method.initialize_testing_set(100)
reduced_basis_method.error_analysis()

# 8. Perform a speedup analysis
reduced_basis_method.initialize_testing_set(100)
reduced_basis_method.speedup_analysis()
