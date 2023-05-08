import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from MixtureOptDesign.MNL.utils import get_i_optimality_bayesian, get_i_optimality_mnl
from sklearn.cluster import AgglomerativeClustering
from typing import List
import pandas as pd




def hierarchical_clustering(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform hierarchical clustering on a dataset and return the mean coordinates of the clusters.

    Parameters
    ----------
    data : ndarray
        An n x d array representing n data points in d dimensions.
    k : int
        The number of clusters to obtain.

    Returns
    -------
    labels : ndarray
        A 1-dimensional ndarray representing the cluster labels of each data point.
    clusters : ndarray
        A k x d ndarray representing the mean coordinates of the data points for each cluster.
    """

    # Compute the pairwise distances between all pairs of data points
    distances = pdist(data)

    # Perform hierarchical clustering on the pairwise distances
    clusters = linkage(distances, method='ward')

    # Extract the labels of each data point based on the number of clusters
    labels = fcluster(clusters, k, criterion='maxclust')

    # Extract the coordinates of the data points for each cluster
    coords = [data[np.where(labels == i)[0], :] for i in range(1, k+1)]

    # Calculate the mean of each cluster
    means = []
    for cluster in coords:
        mean_x = np.mean(cluster[:, 0])
        mean_y = np.mean(cluster[:, 1])
        mean_z = 1 - (mean_x + mean_y)
        means.append(np.array([mean_x, mean_y, mean_z]))

    # Stack the mean coordinates of each cluster
    clusters = np.stack(means)

    return labels, clusters


def replace_with_clusters(data: np.ndarray, labels: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Replace the points in a data array with the corresponding cluster values based on the labels.

    Parameters
    ----------
    data : ndarray
        An n x d array representing n data points in d dimensions.
    labels : ndarray
        An n x 1 array representing the cluster labels of the data points.
    clusters : ndarray
        A k x 1 array representing the cluster values.

    Returns
    -------
    new_data : ndarray
        An n x d array representing the data points with cluster values.

    """

    # Create an empty array to store the new values
    new_data = np.empty_like(data)

    # Iterate through each point in the data array
    for i in range(data.shape[0]):
        # Replace the point with the corresponding cluster value based on the label
        new_data[i, :] = clusters[labels[i]-1]

    return new_data


class Cluster:
    def __init__(self, design:np.ndarray,bayesian:bool=True):
        self._q, self._j, self._s = design.shape
        self._design = design
        self._design_2d = self._design.reshape(self._q, self._j * self._s).T
        
        self.labels = None
        self.clusters = None
    
            
        self.get_i_optimality= get_i_optimality_bayesian if bayesian else get_i_optimality_mnl

    #abstract method
    def fit(self, k:int)-> np.ndarray:
        pass

    
    
    def clustered_design(self)-> np.ndarray:
        if self.clusters is None:
            raise ValueError("No clusters found. Call 'fit' method first.")

        # Use the replace_with_clusters function to assign each data point to its corresponding cluster
        replaced_data = replace_with_clusters(self._design_2d, self.labels, self.clusters)
        return replaced_data.T.reshape(self._design.shape)
        
    
    def get_elbow_curve(self,beta:np.ndarray,order:int,name,linkage_methods:List[str]=['ward', 'complete', 'average']):
        """
        Plots the elbow curve for the given clustering algorithm between the start and end values of k.

        Parameters
        ----------
        beta : numpy.ndarray of shape (,p)
                A 2-dimensional array of p numbers of beta coefficients for the model.
            
        order : int
                The maximum order of interactions to include in the model. Must be 1, 2 or 3.
        linkage_methods : List of str, optional
                The different linkage methods. Default is ['ward', 'complete', 'average']   
            

        Returns
        -------
        plot

        Notes
        -----
        The function computes the I optimality for each value of k between start (number of parameters) and end (unique point) and plots it.
        """
        colors = ['r', 'g', 'b']
        
        # Define the number of clusters to compare
        min_clusters = beta.shape[1] + 1
                
        # Set the maximum number of clusters
        max_clusters = self.get_unique_rows(self._design_2d)
        
        # Create an empty DataFrame to store the I-optimality values
        df = pd.DataFrame(columns=['Method'] + list(range(min_clusters, max_clusters+1)))
        
        for i, linkage_method in enumerate(linkage_methods):
            # Compute the I-optimality criterion for each value of k
            i_opt_values = []
            

            for k in range( min_clusters, max_clusters + 1):
                # Fit the clustering algorithm for the given value of k
                
                
                # Replace the data points with cluster values
                cluster_design =self.fit(k,linkage_method)
                
                # Calculate the I-optimality criterion for the replaced values

                # Calculate the average I-optimality criterion over all Halton draws
                i_opt_avg = self.get_i_optimality(cluster_design, order, beta)

                # Store the I-optimality criterion value
                i_opt_values.append(i_opt_avg)


            # Plot the elbow curve
            plt.plot(range( min_clusters, max_clusters + 1), i_opt_values, colors[i]+'x-',label=linkage_method)
            
            # Add the I-optimality values to the DataFrame
            df.loc[len(df)] = [linkage_method] + i_opt_values
        
        # DataFrame
        df.T.to_csv(f'MixtureOptDesign/data/cluster_{name}.csv', index=False)   
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('I-optimality')
        plt.title('Elbow curve')
        plt.legend()
        
        
        plt.savefig(f'MixtureOptDesign/data/elbow_curve_{name}.png')

        
        plt.show()
    
        
    def get_unique_rows(self,arr:np.ndarray, tolerance=1e-9)->int:
        """
        Get the unique rows of a 2D numpy array based on a specified tolerance level for element-wise equality comparisons.

        Parameters
        ----------
        arr : numpy.ndarray
            The 2D numpy array to get the unique rows of.
        tolerance : float, optional
            The tolerance level for element-wise equality comparisons. Default is 1e-9.

        Returns
        -------
        int
            The length of the unique rows of the input array.

        """
        # Use numpy.isclose to compare the rows with a tolerance level
        close_arr = np.isclose(arr[:, None, :], arr, rtol=tolerance, atol=tolerance).all(axis=2)

        # Use numpy.unique to select the unique rows
        unique_rows_idx = np.unique(close_arr, axis=0, return_index=True)[1]
        unique_rows_idx.sort()
        unique_rows = arr[unique_rows_idx]

        return unique_rows.shape[0]



class HierarchicalCluster(Cluster):
    def fit(self, k:int,linkage_method:str='ward') -> np.ndarray:
        # Compute pairwise distances between all pairs of data points
        distances = pdist(self._design_2d)

        # Perform hierarchical clustering on the pairwise distances using the specific method
        self.clusters = linkage(distances, method=linkage_method)

        # Extract the labels of each data point based on the number of clusters
        self.labels = fcluster(self.clusters, k, criterion='maxclust')

        # Extract the coordinates of the data points for each cluster
        coords = [self._design_2d[np.where(self.labels == i)[0], :] for i in range(1, k+1)]

        # Calculate the mean of each cluster
        means = []
        for cluster in coords:
            # Compute the mean of each dimension separately and append to means list
            if len(cluster) == 0:
                means.append(np.zeros(self._design_2d.shape[1]))
            else:
                mean_x = np.mean(cluster[:, 0])
                mean_y = np.mean(cluster[:, 1])
                mean_z = 1 - (mean_x + mean_y)
                means.append(np.array([mean_x, mean_y, mean_z]))
            

        # Stack the mean coordinates of each cluster
        self.clusters = np.stack(means)
        return self.clustered_design()
        
    def dendogram(self):
        
        dendrogram(self.clusters)
        plt.show()


class AgglomerativeCluster(Cluster):
    def fit(self, k:int) -> np.ndarray:
        # Create an AgglomerativeClustering object with the desired number of clusters
        agglomerative = AgglomerativeClustering(n_clusters=k)

        # Fit the AgglomerativeClustering model to your data
        agglomerative.fit(self._design_2d)

        # Get the cluster labels and clusters from the AgglomerativeClustering object
        labels = agglomerative.labels_
        clusters = [self._design_2d[labels == label] for label in np.unique(labels)]

        # Calculate the mean coordinates of each cluster ensuring they sum up to one
        means = []
        for cluster in clusters:
            mean_x = np.mean(cluster[:, 0])
            mean_y = np.mean(cluster[:, 1])
            mean_z = 1 - (mean_x + mean_y)
            means.append(np.array([mean_x, mean_y, mean_z]))

        # Stack the mean coordinates of each cluster
        centroids = np.stack(means)

        # Create a new numpy array with the same data ordering as the original but with cluster assignments
        ordered_data = np.zeros_like(self._design_2d)
        for label in np.unique(labels):
            ordered_data[labels == label] = centroids[label]

        # Return the ordered data
        return ordered_data


    
    
