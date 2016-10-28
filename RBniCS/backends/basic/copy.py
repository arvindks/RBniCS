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
## @file
#  @brief
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

def copy(arg, backend, wrapping):
    assert isinstance(arg, (backend.Function.Type(), backend.Vector.Type(), backend.Matrix.Type()))
    if isinstance(arg, backend.Function.Type()):
        return wrapping.function_copy(arg)
    elif isinstance(arg, (backend.Matrix.Type(), backend.Vector.Type())):
        return wrapping.tensor_copy(arg)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in copy.")
        
