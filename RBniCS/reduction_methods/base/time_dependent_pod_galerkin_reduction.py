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
## @file elliptic_coercive_reduction_method.py
#  @brief Implementation of projection based reduced order models for elliptic coervice problems: base class
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from math import sqrt
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.io import ErrorAnalysisTable

def TimeDependentPODGalerkinReduction(DifferentialProblemReductionMethod_DerivedClass):
    @Extends(DifferentialProblemReductionMethod_DerivedClass, preserve_class_name=True)
    class TimeDependentPODGalerkinReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        
        ###########################     ERROR ANALYSIS     ########################### 
        ## @defgroup ErrorAnalysis Error analysis
        #  @{
            
        # Compute the error of the reduced order approximation with respect to the full order one
        # over the testing set
        @override
        def error_analysis(self, N=None, **kwargs):
            if "components" in kwargs:
                components = kwargs["components"]
            else:
                components = self.truth_problem.components
                
            time_quadrature = TimeQuadrature((0., self.truth_problem.T), self.truth_problem.dt)
                
            for component in components:
                def solution_preprocess_setitem(component):
                    def solution_preprocess_setitem__function(list_over_time):
                        list_squared_over_time = [v**2 for v in list_over_time]
                        return sqrt(self.time_quadrature.integrate(list_squared_over_time))
                    return solution_preprocess_setitem__function
                for column_prefix in ("error_", "relative_error_"):
                    ErrorAnalysisTable.preprocess_setitem(column_prefix + component, solution_preprocess_setitem(component))
                
            def output_preprocess_setitem(list_over_time):
                return self.time_quadrature.integrate(list_over_time)
            for column in ("error_output", "relative_error_output"):
                ErrorAnalysisTable.preprocess_setitem(column, solution_preprocess_setitem)
                
            DifferentialProblemReductionMethod_DerivedClass.error_analysis(self, N, **kwargs)
            
            ErrorAnalysisTable.clear_setitem_preprocessing()
        
        #  @}
        ########################### end - ERROR ANALYSIS - end ########################### 
        
    # return value (a class) for the decorator
    return TimeDependentPODGalerkinReduction_Class
    