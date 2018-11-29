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

import types
from rbnics.utils.decorators import overload

def compute_theta_for_stability_factor(compute_theta):
    from rbnics.problems.elliptic import EllipticCoerciveProblem, EllipticProblem
    
    module = types.ModuleType("compute_theta_for_stability_factor", "Storage for implementation of compute_theta_for_stability_factor")
    
    def compute_theta_for_stability_factor_impl(self, term):
        return module._compute_theta_for_stability_factor_impl(self, term)
        
    # Elliptic coercive problem
    @overload(EllipticCoerciveProblem, str, module=module)
    def _compute_theta_for_stability_factor_impl(self_, term):
        if term == "stability_factor_left_hand_matrix":
            return tuple(0.5*t for t in compute_theta(self_, "a"))
        else:
            return compute_theta(self_, term)
            
    # Elliptic (non-coercive) problem
    @overload(EllipticProblem, str, module=module)
    def _compute_theta_for_stability_factor_impl(self_, term):
        if term == "stability_factor_left_hand_matrix":
            def Z(theta, p, q):
                return theta[p]*theta[q]

            theta_a = compute_theta(self_, "a")
            Q_a = len(theta_a)
            theta_z = list()
            for p in range(Q_a):
                theta_z.append(sum(- Z(theta_a, p, q) if q != p else Z(theta_a, p, q) for q in range(Q_a)))
            for p in range(Q_a):
                for q in range(p + 1, Q_a):
                    theta_z.append(Z(theta_a, p, q))
            return tuple(theta_z)
        else:
            return compute_theta(self_, term)
    
    return compute_theta_for_stability_factor_impl
