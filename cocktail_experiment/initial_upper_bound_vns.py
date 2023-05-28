import numpy as np
from MixtureOptDesign import HierarchicalCluster, HaltonDraws,plot_ternary_design

import time
import multiprocessing

from MixtureOptDesign import generate_simplex_lattice_design, generate_initial_design

from MixtureOptDesign.vns.vns_cython import vns



def main():
    
    data = np.genfromtxt("Tests/data/beta_mario.csv", delimiter=',')
    beta_ma = np.array(data)
    n_ingredients  = 3
    
    
    
    lattice_design = generate_simplex_lattice_design( n_ingredients,20)
    np.random.seed(10)
    arg1 = list()
    arg2 = list()
    for _ in range(1):
        initial_design, _, other_points = generate_initial_design(lattice_design)
        arg1.append((initial_design,other_points,beta_ma))
        arg2.append(other_points)
           
        
    # create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # record the start time
        start_time = time.perf_counter()

        # use the pool to apply the function to each item in the list in parallel
        results= pool.starmap(vns,arg1)
        
        
        
    # Use max() with a lambda function to retrieve the tuple with the highest third element
    min_tuple = min(results, key=lambda x: x[1])

    # Retrieve the first and second elements from the max_tuple
    first_value, second_value = min_tuple

    print(min_tuple)  # Output: 7 8
        
    end_time = time.perf_counter()
    print("Time taken:", end_time - start_time)
    
if __name__ == '__main__':
    main()
