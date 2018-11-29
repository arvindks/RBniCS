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
from numpy import zeros
from ufl import Form
from dolfin import adjoint, Constant, TestFunction, TrialFunction, split
from rbnics.backends.dolfin.wrapping.dirichlet_bc import DirichletBC
from rbnics.backends.dolfin.wrapping.expression_replace import replace
from rbnics.utils.decorators import overload

def assemble_operator_for_stability_factor(assemble_operator):
    from rbnics.problems.elliptic import EllipticCoerciveProblem, EllipticProblem
    
    module = types.ModuleType("assemble_operator_for_stability_factor", "Storage for implementation of assemble_operator_for_stability_factor")
    
    def assemble_operator_for_stability_factor_impl(self, term):
        return module._assemble_operator_for_stability_factor_impl(self, term)
        
    # Elliptic coercive problem
    @overload(EllipticCoerciveProblem, str, module=module)
    def _assemble_operator_for_stability_factor_impl(self_, term):
        if term == "stability_factor_left_hand_matrix":
            return tuple(f + adjoint(f) for f in assemble_operator(self_, "a"))
        elif term == "stability_factor_right_hand_matrix":
            return assemble_operator(self_, "inner_product")
        elif term == "stability_factor_dirichlet_bc":
            return _homogenize_dirichlet_bcs(assemble_operator(self_, "dirichlet_bc"))
        else:
            return assemble_operator(self_, term)
            
    # Elliptic (non-coercive) problem
    @overload(EllipticProblem, str, module=module)
    def _assemble_operator_for_stability_factor_impl(self_, term):
        if term == "stability_factor_left_hand_matrix":
            operator_a = assemble_operator(self_, "a")
            Q_a = len(operator_a)
            inner_product = assemble_operator(self_, "inner_product")
            assert len(inner_product) is 1
            inner_product = inner_product[0]
            operator_stability_factor_lhs = list()
            for p in range(Q_a):
                operator_stability_factor_lhs.append([[inner_product, operator_a[p]], [adjoint(operator_a[p]), 0]])
            for p in range(Q_a):
                for q in range(p + 1, Q_a):
                    operator_stability_factor_lhs.append([[inner_product, operator_a[p] + operator_a[q]], [adjoint(operator_a[p] + operator_a[q]), 0]])
            return _flatten_block_operator(operator_stability_factor_lhs, self_.V)
        elif term == "stability_factor_right_hand_matrix":
            inner_product = assemble_operator(self_, "inner_product")
            assert len(inner_product) is 1
            return _flatten_block_operator([[0, 0], [0, inner_product]], self_.V, self_.stability_factor_V)
        elif term == "stability_factor_dirichlet_bc":
            homogenized_dirichlet_bcs = _homogenize_dirichlet_bcs(assemble_operator(self_, "dirichlet_bc"))
            return _flatten_block_dirichlet_bc([[homogenized_dirichlet_bc, homogenized_dirichlet_bc] for homogenized_dirichlet_bc in homogenized_dirichlet_bcs], self_.V, self_.stability_factor_V)
        else:
            return assemble_operator(self_, term)
            
    return assemble_operator_for_stability_factor_impl
    
def _homogenize_dirichlet_bcs(original_dirichlet_bcs):
    homogenized_dirichlet_bcs = list()
    for original_dirichlet_bc in original_dirichlet_bcs:
        homogenized_dirichlet_bc = list()
        for original_dirichlet_bc_i in original_dirichlet_bc:
            args = list()
            args.append(original_dirichlet_bc_i.function_space())
            zero_value = Constant(zeros(original_dirichlet_bc_i.value().ufl_shape))
            args.append(zero_value)
            args.extend(original_dirichlet_bc_i._domain)
            kwargs = original_dirichlet_bc_i._kwargs
            homogenized_dirichlet_bc.append(DirichletBC(*args, **kwargs))
        assert len(homogenized_dirichlet_bc) is len(original_dirichlet_bc)
        homogenized_dirichlet_bcs.append(homogenized_dirichlet_bc)
    assert len(homogenized_dirichlet_bcs) is len(original_dirichlet_bcs)
    return tuple(homogenized_dirichlet_bcs)

def _flatten_block_operator(block_operator, V, VV):
    assert V.mesh() == VV.mesh()
    assert VV.num_sub_elements() is 2
    assert V.ufl_element() == VV.sub_elements()[0]
    assert V.ufl_element() == VV.sub_elements()[1]
    v = TestFunction(V)
    vv = split(TestFunction(VV))
    u = TrialFunction(V)
    uu = split(TrialFunction(VV))
    flattened_block_operator = list()
    for op in block_operator:
        flattened_op = 0
        replacements = dict()
        assert len(op) is 2
        assert len(op[0]) is 2
        assert len(op[1]) is 2
        for i in range(2):
            replacements[v] = vv[i]
            for j in range(2):
                replacements[u] = uu[j]
                assert isinstance(op[i][j], Form) or op[i][j] is 0
                if isinstance(op[i][j], Form):
                    flattened_op += replace(op[i][j], replacements)
        assert flattened_op is not 0
        flattened_block_operator.append(flattened_op)
    return tuple(flattened_block_operator)
    
def _flatten_block_dirichlet_bc(block_dirichlet_bcs, V, VV):
    assert V.mesh() == VV.mesh()
    assert VV.num_sub_elements() is 2
    assert V.ufl_element() == VV.sub_elements()[0]
    assert V.ufl_element() == VV.sub_elements()[1]
    flattened_dirichlet_bcs = list()
    for block_dirichlet_bc in block_dirichlet_bcs:
        flattened_dirichlet_bc = list()
        assert len(block_dirichlet_bc) is 2
        flattened_dirichlet_bc = list()
        for i in range(2):
            for block_dirichlet_bc_i in block_dirichlet_bc[i]:
                args = list()
                VV_i = VV.sub(i)
                for c in block_dirichlet_bc_i.function_space().component():
                    VV_i = VV_i.sub(c)
                args.append(VV_i)
                args.append(block_dirichlet_bc_i.value())
                args.extend(block_dirichlet_bc_i._domain)
                kwargs = block_dirichlet_bc_i._kwargs
                flattened_dirichlet_bc.append(DirichletBC(*args, **kwargs))
        assert len(flattened_dirichlet_bc) is len(block_dirichlet_bc[0]) + len(block_dirichlet_bc[1])
        flattened_dirichlet_bcs.append(flattened_dirichlet_bc)
    assert len(flattened_dirichlet_bcs) is len(block_dirichlet_bcs)
    return tuple(flattened_dirichlet_bcs)
