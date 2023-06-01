""""
4.1 Example Description
4.2 Hierarchical clustering of I-optimal design
4.3 Modified coordinate-exchange algorithm
"""
import numpy as np
from MixtureOptDesign import HierarchicalCluster,plot_ternary_design,ClusteredCoordinateExchangeIOptimal
from MixtureOptDesign.data.csv_util import read_csv_file
import time
import multiprocessing
from tabulate import tabulate
import pandas as pd


from MixtureOptDesign.vns.vns_cython import unique_rows

# from MixtureOptDesign.CoordinateExchange.cluster_coordinate_exchange import ClusteredCoordinateExchangeIOptimal


# Time taken: 340.76520660000006s(5.6 min) 6 cores

def main():
    # load beta halton draws
    data = np.genfromtxt("Tests/data/beta_mario.csv", delimiter=',')
    beta_ma = np.array(data)
    
    
    
    
    best_design_mario = read_csv_file("Tests/data/design_03.csv")
    # initial_design = read_csv_file("Tests/data/initial_design_Mario_cocktail.csv")
    
    print(unique_rows(best_design_mario))
    
    #beta = np.array( (1.36, 1.57, 2.47, -0.43, 0.50, 1.09))

    
        
    # fig_initial_design = plot_ternary_design(initial_design)
    # fig_best_design = plot_ternary_design(best_design_mario)
    
    # fig_initial_design.write_image("cocktail_experiment/images/initial_design_cocktail.png")
    # fig_best_design.write_image("cocktail_experiment/images/best_design_cocktail.png")
    
    # halt1 = HaltonDraws(beta,sigma0,128)
    # beta_draws = halt1.generate_draws()
    
    coord = ClusteredCoordinateExchangeIOptimal(num_ingredient=3, num_sets=2, num_choices=16, order=3, n_points=30, bayesian=True, beta=beta_ma, iteration=10, kappa=1, sigma=None)
    
    h_clust = HierarchicalCluster(best_design_mario,)

    h_clust.get_elbow_curve(beta_ma,3,"cocktail_experiment")
    
    # Define the cluster numbers to loop through
    cluster_nums = [13,12,11,10,9,8,7]
    
    cluster_metrics = ["average","complete"]
    for cluster_metric in cluster_metrics:
        
        start_time = time.perf_counter()
        # Loop through the cluster numbers
        
        cluster_design = [("design with {} clusters".format(i),  h_clust.fit(i, cluster_metric)) for i in cluster_nums]
        # print(cluster_design)
        
        # create a pool of worker processes
        with multiprocessing.Pool() as pool:
            # record the start time
            start_time = time.perf_counter()

            # use the pool to apply the function to each item in the list in parallel
            results= pool.map(coord.optimize_design,cluster_design)
            
        end_time = time.perf_counter()
        print("Time taken:", end_time - start_time)
            
        for i in results:
            optimal_design_clust = i[0]
            clust_optimal_design = i[4]
            #print(unique_rows(optimal_design_clust))
            fig_optimal_design = plot_ternary_design(optimal_design_clust,i[2])
            fig_optimal_design.write_image(f"cocktail_experiment/images/{cluster_metric}/after_design_cocktail_{i[1]}.png")
            fig_optimal_design = plot_ternary_design(clust_optimal_design,i[3])
            fig_optimal_design.write_image(f"cocktail_experiment/images/{cluster_metric}/clust_design_cocktail_{i[1]}.png")
            

            # Create a dataframe
        table_data = pd.DataFrame({
            "Cluster Number": cluster_nums, 
            "original I-Optimality value": [i[3] for i in results],
            "Final I-Optimality value":[i[2] for i in results]
        })

        # Generate LaTeX table with headers and without index column
        latex_table = tabulate(table_data, headers='keys', tablefmt="latex_booktabs", 
                            floatfmt=("",".6f",".6f"), showindex=False)

        # Add caption and label
        latex_table = "\\begin{table}[htbp]\n\\centering\n" + latex_table + \
                    "\n\\caption{Optimality criterion values for different cluster numbers before and after coor-dinate exchange with average linkage}\n\\label{tab:Optimality criterion values for different cluster numbers before and after coor-dinate exchange with average linkage}\n\\end{table}"

        # Save LaTeX table to file
        with open(f'cocktail_experiment/tables/coordinate_cluster_{cluster_metric}.tex', "w") as f:
            f.write(latex_table)

if __name__ == '__main__':
    main()

    