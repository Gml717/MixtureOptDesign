""""
4.4 Design construction with an initial upper bound
Design construction with the CE algorithm

"""
import numpy as np
from MixtureOptDesign import plot_ternary_design,ClusteredCoordinateExchangeIOptimal,generate_initial_design
import time
import multiprocessing


from MixtureOptDesign.vns.vns_cython import unique_rows

#Time taken: 7101.8368363s

def main():
    data = np.genfromtxt("Tests/data/beta_mario.csv", delimiter=',')
    beta_ma = np.array(data)

    coord = ClusteredCoordinateExchangeIOptimal(num_ingredient=3, num_sets=2, num_choices=16, order=3, n_points=30, bayesian=True, beta=beta_ma, iteration=10, kappa=1, sigma=None)


    num_mix = 10
    n_ingredients  = 3
    distinct_mix = []
    list_init = []

    np.random.seed(10)

    for _ in range(80):
        
        random_values = np.random.rand(num_mix,n_ingredients)
        design = random_values / np.sum(random_values, axis=1).reshape(num_mix,1)
        
        
        
        distinct_mix.append(design)


    for count, design, in enumerate(distinct_mix):
        init,_,_  =  generate_initial_design(design,k=num_mix)
        
        list_init.append((count,init))
        
            
        
    # create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # record the start time
        start_time = time.perf_counter()

        # use the pool to apply the function to each item in the list in parallel
        results= pool.map(coord.optimize_design,list_init)
        
        
        
    # Use max() with a lambda function to retrieve the tuple with the highest third element
    best_design = min(results, key=lambda x: x[2])

    # Retrieve the first and second elements from the max_tuple
    # first_value, second_value, _ = min_tuple

    print(best_design )  #
    
        
    end_time = time.perf_counter()
    print("Time taken:", end_time - start_time)
    
    fig_optimal_design = plot_ternary_design(best_design [0],best_design [2])
    fig_optimal_design.write_image(f"cocktail_experiment/images/coord_upper_bound_10.png")
    
if __name__ == '__main__':
    main()