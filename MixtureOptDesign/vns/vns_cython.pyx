
import numpy as np
cimport numpy as np


from MixtureOptDesign.MNL.utils import get_i_optimality_bayesian

cpdef vns(np.ndarray[np.float64_t, ndim=3] initial_design, np.ndarray[np.float64_t, ndim=2] other_points,  np.ndarray[np.float64_t, ndim=2] beta):
    cdef np.ndarray[np.double_t, ndim=3] new_design
    cdef double i_opt_value
    cdef int current_neighborhood = 0
    cdef int num_neighborhoods = 2
    
    while current_neighborhood < num_neighborhoods:
        
        # Explore current neighborhood
        if current_neighborhood == 0:
            new_design, i_opt_value, improvement = neighborhood_func_3(other_points, initial_design, beta)
        elif current_neighborhood == 1:
            new_design, i_opt_value, improvement = cython_neighborhood_func_1(other_points, initial_design, beta)
        else:
            new_design, i_opt_value, improvement = neighborhood_func_2_cython(other_points, initial_design, beta)
        
        # Check if improvement was found
        if improvement:
            current_neighborhood = 0
        else:
            current_neighborhood += 1
        
        initial_design = new_design

    print(initial_design)
    print(i_opt_value)
            
    return initial_design, i_opt_value







cdef  cython_neighborhood_func_1(np.ndarray[np.float64_t, ndim=2] other_points,np.ndarray[np.float64_t, ndim=3] initial_design, np.ndarray[np.float64_t, ndim=2] beta):
    
    cdef np.ndarray[np.float64_t, ndim=3] canditate_design
    cdef np.ndarray[np.float64_t, ndim=2] rows
    cdef double i_opt_value
    cdef double i_new_value
    cdef bint improvement
    cdef int s, j, k, p
    
    _, alternatives, choice = np.shape(initial_design)
    i_opt_value = get_i_optimality_bayesian(initial_design, 3, beta)
    rows = unique_rows(initial_design)
    
    while improvement:
        improvement = False
        for s in range(choice):
            for j in range(alternatives):
                
                canditate_design = initial_design.copy()
                # Get the current mixture
                current_mix = initial_design[:, j, s]
                # Iterate over each unique point in the initial_design
                for k in range(rows.shape[0]):
                    unique_points = rows[k, :]
                    # Skip the value of the current point
                    if np.array_equal(unique_points, current_mix):
                        continue
                    canditate_design = initial_design.copy()
                    # Replace the current point with the unique point
                    canditate_design[:, j, s] = unique_points.copy()

                    # Compute the I-optimality of the candidate initial_design
                    i_new_value = get_i_optimality_bayesian(canditate_design, 3, beta)

                    # If the I-optimality is improved, update the initial_design
                    if i_new_value != 100 and  i_opt_value > i_new_value and abs((i_opt_value - i_new_value)) > 0.01:
                        initial_design = canditate_design
                        i_opt_value = i_new_value
                        improvement = True
                        break
                if improvement:
                    break
                
            if improvement:
                break
        
        #if improvement:
            #initial_design, i_opt_value = neighborhood_func_1( other_points, initial_design, beta)
        
    return initial_design, i_opt_value,improvement
        
    


def unique_rows(np.ndarray[np.float64_t, ndim=3] design):
    cdef int q,j,s
    cdef np.ndarray[np.float64_t, ndim=2] arr
    q,j,s = np.shape(design)
    arr = design.T.reshape(j*s,q)
    return np.unique(arr,axis=0)



# Define the Cython function with the decorator
cdef neighborhood_func_2_cython(np.ndarray[np.float64_t, ndim=2] other_points, np.ndarray[np.float64_t, ndim=3] initial_design, np.ndarray[np.float64_t, ndim=2] beta):

    cdef int _, alternatives, choice,choice_idx,alternative_idx,other_choice_idx
    cdef double i_opt_value, i_new_value
    cdef bint improvement
    cdef np.ndarray[np.float64_t, ndim=3] canditate_design

    _, alternatives, choice = np.shape(initial_design)
    i_opt_value = get_i_optimality_bayesian(initial_design, 3, beta)
    improvement = False
    
    for choice_idx in range(choice):
        for alternative_idx in range(alternatives):
                for other_choice_idx in range(choice):
                    for other_alternative_idx in range(alternatives):
                        improvement = False
                        if choice_idx == other_choice_idx :
                            continue
                        canditate_design = initial_design.copy()
                        canditate_design[:, alternative_idx, choice_idx], canditate_design[:, other_alternative_idx, other_choice_idx] = canditate_design[:, other_alternative_idx, other_choice_idx], canditate_design[:, alternative_idx, choice_idx]
                        i_new_value = get_i_optimality_bayesian(canditate_design, 3, beta)

                        if i_new_value != 100 and  i_opt_value > i_new_value and abs((i_opt_value - i_new_value)) > 0.01:
                            initial_design = canditate_design
                            i_opt_value = i_new_value
                            improvement = True
                            break
                    if improvement:
                        break
                if improvement:
                    break
                
        if improvement:
            break

    return initial_design, i_opt_value, improvement





cdef  neighborhood_func_3(np.ndarray[np.float64_t,ndim=2] other_points, np.ndarray[np.float64_t, ndim=3] initial_design, np.ndarray[np.float64_t, ndim=2] beta):
    cdef np.ndarray[np.float64_t, ndim=2] design_points = unique_rows(initial_design)
    cdef int q = design_points.shape[1]
    #cdef np.ndarray[np.float64_t ,ndim=1] indices
    cdef tuple indices
    cdef np.ndarray[np.float64_t,ndim=3] candidate_design
    cdef double i_opt_value = get_i_optimality_bayesian(initial_design, 3, beta)
    cdef double i_new_value
    cdef bint improvement = False
    cdef int i,j
    
    for i in range(design_points.shape[0]):
        indices = np.where(np.all(initial_design == design_points[i].reshape(q, 1, 1), axis=0))
        for j in range(other_points.shape[0]):
            candidate_design = initial_design.copy()
            candidate_design[:, indices[0], indices[1]] = other_points[j].reshape(q, 1).copy()
            i_new_value = get_i_optimality_bayesian(candidate_design, 3, beta)
            if i_new_value != 100  and i_opt_value > i_new_value and abs(i_opt_value - i_new_value) > 0.01:
                initial_design = candidate_design
                i_opt_value = i_new_value
                other_points[j] = design_points[i].copy()
                    
            
        
        
    return initial_design, i_opt_value, improvement


