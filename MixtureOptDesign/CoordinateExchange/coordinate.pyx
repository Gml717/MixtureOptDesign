import numpy as np
cimport numpy as np
import logging



from MixtureOptDesign.MNL.utils import *

from MixtureOptDesign.HaltonDraws.halton_draws import HaltonDraws

from MixtureOptDesign.CoordinateExchange.coordinate_exchange import unique_rows



cdef class CoordinateExchangeIOptimal:
    cdef public np.ndarray design
    cdef public int order
    cdef public int n_points
    cdef public int iteration
    cdef public np.ndarray beta
    cdef public int num_ingredient
    cdef public int num_choices
    cdef public int num_sets
    cdef public np.ndarray sigma
    cdef public int kappa
    cdef public float i_opt_value
    cdef public np.npy_bool bayesian
    cdef public int num_param
      

    def __init__(
        self,
        int num_ingredient,
        int num_sets,
        int num_choices,
        int order, 
        int n_points,
        int iteration=10,
        np.ndarray[np.double_t, ndim=2] design=None, 
        beta=None,#check this later for improvement
        np.npy_bool bayesian=True,
        np.ndarray[np.double_t, ndim=2] sigma=None,
        int kappa=1
        ):
        
        self.design = design
        self.order = order
        self.n_points = n_points
        self.iteration = iteration
        self.num_ingredient = num_ingredient
        self.num_sets = num_sets
        self.num_choices = num_choices
        self.num_param = sum(get_parameters(num_ingredient, order)) + 1





        
        if beta is None:
            self.beta = generate_beta_params(self.num_param, num_ingredient)
        else:
            self.beta = beta

        
        if bayesian and self.beta.ndim == 1:
            
            if sigma is None:
                self.sigma = transform_varcov_matrix(np.identity(self.beta.size + 1), q=num_ingredient, k=kappa)
            else:
                self.sigma = sigma
            
            self.beta = HaltonDraws(self.beta, self.sigma, 128).generate_draws()
    

    # define the function in Cython
    cpdef np.ndarray optimize_design(self,  design_numer):
        # define the input parameters
        cdef int n_ingredients = self.num_ingredient
        cdef int n_sets = self.num_sets
        cdef int n_choices = self.num_choices
        cdef int n_points = self.n_points
        cdef int iteration = self.iteration
        cdef float opt_crit_value_orig, i_best, i_opt_critc_value, i_new_value
        cdef int j, s, q, cox_direction, it
        cdef np.ndarray[np.float_t, ndim=3] design, candidate_design
        cdef np.ndarray[np.float_t, ndim=2] cox_directions

        # initialize the design with random initial design
        design = get_random_initial_design_mnl(n_ingredients=n_ingredients,n_alternatives=n_sets,n_choice_sets=n_choices,seed=design_numer[1])

        # set up initial optimality value   
        opt_crit_value_orig = get_i_optimality_bayesian(design, self.order, self.beta)
        i_best = opt_crit_value_orig
        i_opt_critc_value = 100

        it = 0
        for _ in range(iteration):

            # If there was no improvement in this iteration
            if abs(i_opt_critc_value - i_best) < 0.01:
                break

            i_opt_critc_value = i_best
            it += 1
            for j in range(n_sets):
                for s in range(n_choices):
                    for q in range(n_ingredients):
                        cox_directions = compute_cox_direction(design[:, j, s], q, n_points)
                        # Loop through each Cox direction
                        for cox_direction in range(cox_directions.shape[0]):
                            # Create candidate design by copying current design
                            candidate_design = design.copy()
                            # Replace the current Mixture with the Cox direction
                            candidate_design[:, j, s] = cox_directions[cox_direction,:]
                            # Compute optimality criterion for candidate design
                            i_new_value = get_i_optimality_bayesian(candidate_design, self.order, self.beta)
                            # Update the design and optimality  if there's an improvement
                            if i_new_value != 100 and  i_best > i_new_value and abs((i_best - i_new_value)) > 0.01:
                                design = candidate_design
                                i_best = i_new_value

        logging.basicConfig(filename='design_optimization.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

        logging.info("Initial design: %s", design_numer[0])
        logging.info("Original optimality criterion value: %s", opt_crit_value_orig)
        logging.info("Final optimality criterion value: %s", i_best)
        logging.info("Number of iterations: %s", it)

        return design


class ClusteredCoordinateExchangeIOptimal(CoordinateExchangeIOptimal):
    
    def optimize_design(self,design_) -> tuple:
        """
        Optimize design in regards to the optimality criterion.

        Returns:
            np.ndarray: Optimized design 
    """
        # define the input parameters
        cdef int n_ingredients = self.num_ingredient
        cdef int n_sets = self.num_sets
        cdef int n_choices = self.num_choices
        cdef int n_points = self.n_points
        cdef int iteration = self.iteration
        cdef float opt_crit_value_orig, i_best, i_opt_critc_value, i_new_value
        cdef int j, s, q, cox_direction, it
        cdef np.ndarray[np.float_t, ndim=3] design, candidate_design
        cdef np.ndarray[np.float_t, ndim=2] cox_directions, unique_design_points 
        


        
        design = design_[1].copy()
        unique_design_points = unique_rows(design.copy())
       
        # set up initial optimality value   
        opt_crit_value_orig = get_i_optimality_bayesian(design,self.order,self.beta)
        i_best = opt_crit_value_orig 
        i_opt_critc_value = 100
        
        
        it = 0
        for _ in range(self.iteration):
            # If there was no improvement in this iteration
            if abs(i_opt_critc_value - i_best) < 0.001:
                break
            
            i_opt_critc_value = i_best
            it += 1
            for i in range(len(unique_design_points)):
                
                    for q in range(self.num_ingredient):
                        cox_directions = compute_cox_direction(unique_design_points[i], q, self.n_points)
                        # Loop through each Cox direction
                        for cox_direction in range(cox_directions.shape[0]):
                            # Create candidate design by copying current design
                            canditate_design = design.copy()
                            indices = np.where(np.all(canditate_design == unique_design_points[i].reshape(self.num_ingredient,1,1), axis=0))
                            subset = canditate_design[:,indices[0],indices[1]].shape
                            cox = cox_directions[cox_direction,:].reshape(self.num_ingredient,1,1)
                            # Replace the current cluster Mixture with the Cox direction
                            canditate_design[:,indices[0],indices[1]] = np.zeros(subset) + cox.reshape(3,1)
                            
                            
                        
                            # Compute optimality criterion for candidate design
                            i_new_value = get_i_optimality_bayesian(canditate_design,self.order,self.beta)
                            if i_new_value != 100 and  i_best > i_new_value and abs((i_best - i_new_value)) > 0.001 :


                                
                                design = canditate_design.copy()
                                
                                unique_design_points[i] = cox.reshape(3,)
                                i_best = i_new_value
                                    
        print(design_[0])
        print("Original Optimality criterion value: ", opt_crit_value_orig)
        print("Final Optimality criterion value: ", i_best)
        print("Number of iterations: ", it)
        return design, design_[0],i_best,opt_crit_value_orig,design_[1]
