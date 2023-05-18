
from MixtureOptDesign.MNL.utils import get_i_optimality_bayesian
import numpy as np

from MixtureOptDesign.vns.vns_cython import *



def vns(initial_design, other_points, beta):
    neighborhoods = [neighborhood_func_1,neighborhood_func_2,neighborhood_func_3]
    num_neighborhoods = len(neighborhoods)
    current_neighborhood = 0
    
    while current_neighborhood < num_neighborhoods:
        
        # Explore current neighborhood
        initial_design, i_opt_value,improvement = neighborhoods[current_neighborhood](other_points, initial_design, beta)
        
        # Check if improvement was found
        if improvement:
            current_neighborhood = 0
        else:
            current_neighborhood += 1
            
    return initial_design, i_opt_value




# def neighborhood_func_3( other_points, initial_design, beta):
#     i_opt_value = get_i_optimality_bayesian(initial_design, 3, beta)
    
    
#     design_points = unique_rows(initial_design)
#     q = design_points.shape[1]
#     for i in range(len(design_points)):
#         improvement = False
#         indices = np.where(np.all(initial_design == design_points[i].reshape(q, 1, 1), axis=0))
#         try:
#             for j in range(len(other_points)):
#                 canditate_design = initial_design.copy()
#                 canditate_design[:, indices[0], indices[1]] = other_points[j].reshape(q, 1).copy()
#                 i_new_value = get_i_optimality_bayesian(canditate_design, 3, beta)
#                 if i_new_value != 100 and  i_opt_value > i_new_value and abs((i_opt_value - i_new_value)) > 0.01:
#                     initial_design = canditate_design
#                     i_opt_value = i_new_value
#                     other_points[j] = design_points[i].copy()
#                     improvement = True
#                     break
#             if improvement:
#                 break
#         except np.linalg.LinAlgError:
#             print("Singular matrix!")
#             continue
        
#     return initial_design, i_opt_value,improvement

# def neighborhood_func_1(other_points, initial_design, beta):
#     _, alternatives, choice = initial_design.shape
#     i_opt_value = get_i_optimality_bayesian(initial_design, 3, beta)
#     rows = unique_rows(initial_design)
#     improvement = True
    
#     while improvement:
#         improvement = False
#         for s in range(choice):
#             for j in range(alternatives):
#                 try:
#                     canditate_design = initial_design.copy()
#                     # Get the current mixture
#                     current_mix = initial_design[:, j, s]
#                     # Iterate over each unique point in the initial_design
#                     for unique_points in rows:
#                         # Skip the value of the current point
#                         if np.array_equal(unique_points, current_mix):
#                             continue
#                         canditate_design = initial_design.copy()
#                         # Replace the current point with the unique point
#                         canditate_design[:, j, s] = unique_points.copy()

#                         # Compute the I-optimality of the candidate initial_design
#                         i_new_value = get_i_optimality_bayesian(canditate_design, 3, beta)

#                         # If the I-optimality is improved, update the initial_design
#                         if i_new_value != 100 and  i_opt_value > i_new_value and abs((i_opt_value - i_new_value)) > 0.01:
#                             initial_design = canditate_design
#                             i_opt_value = i_new_value
#                             improvement = True
#                             break
#                     if improvement:
#                         break
#                 except np.linalg.LinAlgError:
#                     print("Singular matrix!")
#                     continue
            
#             if improvement:
#                     break
                    
#         #if improvement:
#             #initial_design, i_opt_value = neighborhood_func_1( other_points, initial_design, beta)
        

#     return initial_design, i_opt_value,improvement



# def neighborhood_func_2(other_points, initial_design, beta):
#     _, alternatives, choice = initial_design.shape
#     i_opt_value = get_i_optimality_bayesian(initial_design, 3, beta)
#     improvement = False
    
#     for choice_idx in range(choice):
#         for alternative_idx in range(alternatives):
#             try:
#                 # Swap the current point with every other point in sequence
#                 for other_choice_idx in range(choice):
#                     for other_alternative_idx in range(alternatives):
#                         improvement = False
#                         if choice_idx == other_choice_idx :
#                             continue
#                         canditate_design = initial_design.copy()
#                         canditate_design[:, alternative_idx, choice_idx], canditate_design[:, other_alternative_idx, other_choice_idx] = canditate_design[:, other_alternative_idx, other_choice_idx], canditate_design[:, alternative_idx, choice_idx]
#                         i_new_value = get_i_optimality_bayesian(canditate_design, 3, beta)
                        
#                         # improvement with a minimum scale of 0.01
#                         # still need to change the scale for the change in i optimality
#                         if i_new_value != 100 and  i_opt_value > i_new_value and abs((i_opt_value - i_new_value)) > 0.01:
#                             initial_design = canditate_design
#                             i_opt_value = i_new_value
#                             improvement = True
#                             break
#                     if improvement:
#                         break
#                 if improvement:
#                     break
#             except np.linalg.LinAlgError:
#                 print("Singular matrix encountered during neighborhood_func_3!")
                
#         if improvement:
#             break


#     return initial_design, i_opt_value, improvement



def unique_rows(design:np.ndarray)->np.ndarray:
    q,j,s = design.shape
    arr = design.T.reshape(j*s,q)
    return np.unique(arr,axis=0)
