from MixtureOptDesign.data.csv_util import read_csv_file
import numpy as np
from MixtureOptDesign import HierarchicalCluster, HaltonDraws,plot_ternary_design
from MixtureOptDesign.data.csv_util import read_csv_file
from MixtureOptDesign.CoordinateExchange.coordinate import ClusteredCoordinateExchangeIOptimal
import time
import multiprocessing

def unique_rows(design:np.ndarray)->np.ndarray:
    q,j,s = design.shape
    arr = design.T.reshape(j*s,q)
    return np.unique(arr,axis=0)


def main():
    # load beta halton draws
    data = np.genfromtxt("Tests/data/beta_mario.csv", delimiter=',')
    beta_ma = np.array(data)
    
    
    best_design_mario = read_csv_file("Tests/data/design_03.csv")
    initial_design = read_csv_file("Tests/data/initial_design_Mario_cocktail.csv")
    
    beta = np.array( (1.36, 1.57, 2.47, -0.43, 0.50, 1.09))

    
        
    fig_initial_design = plot_ternary_design(initial_design)
    fig_best_design = plot_ternary_design(best_design_mario)
    
    fig_initial_design.write_image("MixtureOptDesign/data/initial_design_cocktail.png")
    fig_best_design.write_image("MixtureOptDesign/data/best_design_cocktail.png")
    
    # halt1 = HaltonDraws(beta,sigma0,128)
    # beta_draws = halt1.generate_draws()
    
    coord = ClusteredCoordinateExchangeIOptimal(num_ingredient=3, num_sets=2, num_choices=16, order=3, n_points=30, bayesian=True, beta=beta_ma, iteration=10, kappa=1, sigma=None)
    
    h_clust = HierarchicalCluster(best_design_mario,)

    h_clust.get_elbow_curve(beta_ma,3,"cocktail")
    
    # Define the cluster numbers to loop through
    cluster_nums = [10,9,8,7]
    
    cluster_metric = "average"
    start_time = time.perf_counter()
    # Loop through the cluster numbers
    
    cluster_design = [("design with {} clusters".format(i+1),  h_clust.fit(i, cluster_metric)) for i in cluster_nums]
    print(cluster_design)
    
    # create a pool of worker processes
    with multiprocessing.Pool() as pool:
        # record the start time
        start_time = time.perf_counter()

        # use the pool to apply the function to each item in the list in parallel
        results= pool.map(coord.optimize_design,cluster_design)
        
        
        
    for i in results:
        optimal_design_clust = i[0]
        print(unique_rows(optimal_design_clust))
        fig_optimal_design = plot_ternary_design(optimal_design_clust)
        fig_optimal_design.write_image(f"MixtureOptDesign/data/{cluster_metric}_design_cocktail_{i[1]}.png")
    
    # for cluster_num in cluster_nums:

        # # Fit the hierarchical clustering model
        # h_clust.fit(cluster_num, cluster_metric )

        # # Get the replaced data and reshape it
        # cluster_design = h_clust.clustered_design()

        # Optimize the design and plot it
        # optimal_design_clust = coord.optimize_design(design_=cluster_design)
        # fig_optimal_design = plot_ternary_design(optimal_design_clust)

        # # Save the plot as an image
        # fig_optimal_design.write_image(f"MixtureOptDesign/data/{cluster_metric}_design_cocktail_{cluster_num}.png")
    end_time = time.perf_counter()
    print("Time taken:", end_time - start_time)



if __name__ == '__main__':
    main()