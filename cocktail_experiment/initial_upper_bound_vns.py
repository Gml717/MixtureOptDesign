"""
4.4 Design construction with an initial upper bound
Design construction with the modified VNS algorithm

"""

import numpy as np
from MixtureOptDesign import plot_ternary_design,generate_initial_design,generate_simplex_lattice_design

import time
import multiprocessing


from MixtureOptDesign.vns.vns_cython import vns

#Time taken: 25992.401746699998s(7.5 hours) 6 cores

def main():
    
    data = np.genfromtxt("Tests/data/beta_mario.csv", delimiter=',')
    beta_ma = np.array(data)
    n_ingredients  = 3
    
    
    
    lattice_design = generate_simplex_lattice_design( n_ingredients,25)
    np.random.seed(10)
    arg1 = list()
    order = 3
    for _ in range(20):
        initial_design, _, other_points = generate_initial_design(lattice_design)
        arg1.append((initial_design,other_points,beta_ma,order))
        

    # create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # record the start time
        start_time = time.perf_counter()

        # use the pool to apply the function to each item in the list in parallel
        results= pool.starmap(vns,arg1)
        
        
        
    # Use max() with a lambda function to retrieve the tuple with lowest I-opt
    vns_design = min(results, key=lambda x: x[1])


    print(vns_design)  # Output: 7 8
        
    end_time = time.perf_counter()
    print("Time taken:", end_time - start_time)
    fig_optimal_design = plot_ternary_design(vns_design[0],vns_design[1])
    fig_optimal_design.write_image(f"cocktail_experiment/images/vns_upper_bound_10.png")
    
if __name__ == '__main__':
    main()
