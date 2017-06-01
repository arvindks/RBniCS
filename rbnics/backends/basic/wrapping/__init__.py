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

from rbnics.backends.basic.wrapping.function_copy import function_copy
from rbnics.backends.basic.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.backends.basic.wrapping.function_load import function_load
from rbnics.backends.basic.wrapping.function_save import function_save
from rbnics.backends.basic.wrapping.functions_list_basis_functions_matrix_adapter import functions_list_basis_functions_matrix_adapter
from rbnics.backends.basic.wrapping.functions_list_basis_functions_matrix_mul import functions_list_basis_functions_matrix_mul_online_matrix, functions_list_basis_functions_matrix_mul_online_vector, functions_list_basis_functions_matrix_mul_online_function
from rbnics.backends.basic.wrapping.get_function_subspace import get_function_subspace
from rbnics.backends.basic.wrapping.get_mpi_comm import get_mpi_comm
from rbnics.backends.basic.wrapping.gram_schmidt_projection_step import gram_schmidt_projection_step
from rbnics.backends.basic.wrapping.matrix_mul import matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix
from rbnics.backends.basic.wrapping.tensor_copy import tensor_copy
from rbnics.backends.basic.wrapping.tensor_load import tensor_load
from rbnics.backends.basic.wrapping.tensor_save import tensor_save
from rbnics.backends.basic.wrapping.tensors_list_mul import tensors_list_mul_online_function
from rbnics.backends.basic.wrapping.vector_mul import vector_mul_vector

__all__ = [
    'function_copy',
    'function_extend_or_restrict',
    'function_load',
    'function_save',
    'functions_list_basis_functions_matrix_adapter',
    'functions_list_basis_functions_matrix_mul_online_matrix',
    'functions_list_basis_functions_matrix_mul_online_vector',
    'functions_list_basis_functions_matrix_mul_online_function',
    'get_function_subspace',
    'get_mpi_comm',
    'gram_schmidt_projection_step',
    'matrix_mul_vector',
    'tensor_copy',
    'tensor_load',
    'tensor_save',
    'tensors_list_mul_online_function',
    'vector_mul_vector',
    'vectorized_matrix_inner_vectorized_matrix'
]