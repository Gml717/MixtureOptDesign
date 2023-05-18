import numpy as np
from MixtureOptDesign.MNL.utils  import compute_cox_direction
from ..HaltonDraws.halton_draws import HaltonDraws

from .coordinate_exchange import CoordinateExchangeIOptimal,unique_rows





class ClusteredCoordinateExchangeIOptimal(CoordinateExchangeIOptimal):
    
    def optimize_design(self,design_) -> np.ndarray:
        """
        Optimize design in regards to the optimality criterion.

        Returns:
            np.ndarray: Optimized design 
    """
        
        design = design_[1].copy()
        unique_design_points = unique_rows(design)
        
        # set up initial optimality value   
        opt_crit_value_orig =  self.get_i_optimality(design,self._order,self._beta)
        i_best = opt_crit_value_orig 
        i_opt_critc_value = float("inf")
        
        
        it = 0
        for _ in range(self._iteration):
            # If there was no improvement in this iteration
            if abs(i_opt_critc_value - i_best) < 0.01:
                break
            
            i_opt_critc_value = i_best
            it += 1
            for i in range(len(unique_design_points)):
                
                    for q in range(self._num_ingredient):
                        cox_directions = compute_cox_direction(unique_design_points[i], q, self._n_points)
                        # Loop through each Cox direction
                        for cox_direction in range(cox_directions.shape[0]):
                            # Create candidate design by copying current design
                            canditate_design = design.copy()
                            indices = np.where(np.all(canditate_design == unique_design_points[i].reshape(self._num_ingredient,1,1), axis=0))
                            subset = canditate_design[:,indices[0],indices[1]].shape
                            cox = cox_directions[cox_direction,:].reshape(self._num_ingredient,1,1)
                            # Replace the current cluster Mixture with the Cox direction
                            canditate_design[:,indices[0],indices[1]] = np.zeros(subset) + cox.reshape(3,1)
                            try:
                                # Compute optimality criterion for candidate design
                                i_new_value = self.get_i_optimality(canditate_design,self._order,self._beta)
                            except np.linalg.LinAlgError:
                                # Skip to the next Cox direction if a LinAlgError occurs
                                continue
                            if i_new_value > 0 and i_best - i_new_value >= 0.01:
                                design = canditate_design.copy()
                                unique_design_points[i] = cox.reshape(3,)
                                i_best = i_new_value
        print("Original Optimality criterion value: ", opt_crit_value_orig)
        print("Final Optimality criterion value: ", i_best)
        print("Number of iterations: ", it)
        return design,design_[0],i_best