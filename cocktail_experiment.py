from MixtureOptDesign.data.csv_util import read_csv_file
import numpy as np
from MixtureOptDesign.MNL.utils import get_i_optimality_bayesian
from MixtureOptDesign import HierarchicalCluster, HaltonDraws,plot_ternary_design
from MixtureOptDesign.data.csv_util import read_csv_file
from MixtureOptDesign.CoordinateExchange.coordinate import ClusteredCoordinateExchangeIOptimal
import time
import multiprocessing
from MixtureOptDesign import generate_initial_design
from tabulate import tabulate
import pandas as pd

from MixtureOptDesign import generate_simplex_lattice_design, generate_initial_design,vns

from MixtureOptDesign.vns.vns_cython import vns

# from MixtureOptDesign.CoordinateExchange.cluster_coordinate_exchange import ClusteredCoordinateExchangeIOptimal

def unique_rows(design:np.ndarray)->np.ndarray:
    q,j,s = design.shape
    arr = design.T.reshape(j*s,q)
    return np.unique(arr,axis=0)


def main():
    # load beta halton draws
    data = np.genfromtxt("Tests/data/beta_mario.csv", delimiter=',')
    beta_ma = np.array(data)
    
    
    best_design_mario = read_csv_file("Tests/data/design_03.csv")
    # initial_design = read_csv_file("Tests/data/initial_design_Mario_cocktail.csv")
    
    beta = np.array( (1.36, 1.57, 2.47, -0.43, 0.50, 1.09))

    
        
    # fig_initial_design = plot_ternary_design(initial_design)
    # fig_best_design = plot_ternary_design(best_design_mario)
    
    # fig_initial_design.write_image("MixtureOptDesign/data/initial_design_cocktail.png")
    # fig_best_design.write_image("MixtureOptDesign/data/best_design_cocktail.png")
    
    # halt1 = HaltonDraws(beta,sigma0,128)
    # beta_draws = halt1.generate_draws()
    
    coord = ClusteredCoordinateExchangeIOptimal(num_ingredient=3, num_sets=2, num_choices=16, order=3, n_points=30, bayesian=True, beta=beta_ma, iteration=10, kappa=1, sigma=None)
    
    # h_clust = HierarchicalCluster(best_design_mario,)

    # h_clust.get_elbow_curve(beta_ma,3,"cocktail")
    
    # # Define the cluster numbers to loop through
    # cluster_nums = [13,12,11,10,9,8,7]
    
    # cluster_metrics = ["average","complete"]
    # for cluster_metric in cluster_metrics:
        
    #     start_time = time.perf_counter()
    #     # Loop through the cluster numbers
        
    #     cluster_design = [("design with {} clusters".format(i),  h_clust.fit(i, cluster_metric)) for i in cluster_nums]
    #     # print(cluster_design)
        
    #     # create a pool of worker processes
    #     with multiprocessing.Pool() as pool:
    #         # record the start time
    #         start_time = time.perf_counter()

    #         # use the pool to apply the function to each item in the list in parallel
    #         results= pool.map(coord.optimize_design,cluster_design)
            
    #     end_time = time.perf_counter()
    #     print("Time taken:", end_time - start_time)
            
    #     for i in results:
    #         optimal_design_clust = i[0]
    #         print(unique_rows(optimal_design_clust))
    #         fig_optimal_design = plot_ternary_design(optimal_design_clust)
    #         fig_optimal_design.write_image(f"MixtureOptDesign/data/images/{cluster_metric}_design_cocktail_{i[1]}.png")
            

    #         # Create a dataframe
    #     table_data = pd.DataFrame({
    #         "Cluster Number": cluster_nums, 
    #         "original I-Optimality value": [i[3] for i in results],
    #         "Final I-Optimality value":[i[2] for i in results]
    #     })

    #     # Generate LaTeX table with headers and without index column
    #     latex_table = tabulate(table_data, headers='keys', tablefmt="latex_booktabs", 
    #                         floatfmt=("",".6f",".6f"), showindex=False)

    #     # Add caption and label
    #     latex_table = "\\begin{table}[htbp]\n\\centering\n" + latex_table + \
    #                 "\n\\caption{Optimality criterion values for different cluster numbers before and after coor-dinate exchange with average linkage}\n\\label{tab:Optimality criterion values for different cluster numbers before and after coor-dinate exchange with average linkage}\n\\end{table}"

    #     # Save LaTeX table to file
    #     with open(f'MixtureOptDesign/data/tables/coordinate_cluster_{cluster_metric}.tex', "w") as f:
    #         f.write(latex_table)
    # def unique_rows(design:np.ndarray)->np.ndarray:
    #     q,j,s = design.shape
    #     arr = design.T.reshape(j*s,q)
    #     return np.unique(arr,axis=0)
    
        
    
    num_mix = 10
    n_ingredients  = 3
    pourbaix = []
    list_init = []
    
    np.random.seed(10)

    for _ in range(40):
        
        random_values = np.random.rand(num_mix,n_ingredients)
        design = random_values / np.sum(random_values, axis=1).reshape(num_mix,1)
        
        
        
        pourbaix.append(design)
    
    i = 0
    
    
    for count, design, in enumerate(pourbaix):
        init,j,p  =  generate_initial_design(design,k=num_mix)
        
        list_init.append((count,init))
        
        

        
        
    # create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # record the start time
        start_time = time.perf_counter()

        # use the pool to apply the function to each item in the list in parallel
        results= pool.map(coord.optimize_design,list_init)
        
        
        
    # Use max() with a lambda function to retrieve the tuple with the highest third element
    min_tuple = min(results, key=lambda x: x[2])

    # Retrieve the first and second elements from the max_tuple
    # first_value, second_value, _ = min_tuple

    print(min_tuple)  # Output: 7 8
    print(unique_rows(min_tuple[0]))
        
    end_time = time.perf_counter()
    print("Time taken:", end_time - start_time)
    
    
    # num_mix = 13
    # n_ingredients  = 3
    
    
    
    # lattice_design = generate_simplex_lattice_design(3,25)
    # np.random.seed(10)
    # arg1 = list()
    # arg2 = list()
    # for _ in range(20):
    #     initial_design, _, other_points = generate_initial_design(lattice_design)
    #     arg1.append((initial_design,other_points,beta_ma))
    #     arg2.append(other_points)
        
   
        
    
        
        
    
        
        
    # # create a pool of worker processes
    # with multiprocessing.Pool() as pool:
    #     # record the start time
    #     start_time = time.perf_counter()

    #     # use the pool to apply the function to each item in the list in parallel
    #     results= pool.starmap(vns,arg1)
        
        
        
    # # Use max() with a lambda function to retrieve the tuple with the highest third element
    # min_tuple = min(results, key=lambda x: x[1])

    # # Retrieve the first and second elements from the max_tuple
    # first_value, second_value = min_tuple

    # print(min_tuple)  # Output: 7 8
        
    # end_time = time.perf_counter()
    # print("Time taken:", end_time - start_time)



if __name__ == '__main__':
    main()