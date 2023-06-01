import multiprocessing
import time
import numpy as np
import cProfile
from MixtureOptDesign.coordinate_exchange.coordinate import CoordinateExchangeIOptimal

def main():
    np.random.seed(42)
    initial_designs = [("design{}".format(i+1), np.random.randint(1, 1e9)) for i in range(10)]

    coord_exchange = CoordinateExchangeIOptimal(num_ingredient=3, num_sets=2, num_choices=16, order=3, n_points=30, bayesian=True, beta=None, iteration=5, kappa=1, sigma=None)
    # create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # record the start time
        start_time = time.perf_counter()

        # use the pool to apply the function to each item in the list in parallel
        results = pool.map(coord_exchange.optimize_design, initial_designs)

        print(results)   
        end_time = time.perf_counter()
        print("Time taken:", end_time - start_time)
        #Time taken: 2371.2889697


if __name__ == '__main__':
    main()
    #cProfile.run('main()')