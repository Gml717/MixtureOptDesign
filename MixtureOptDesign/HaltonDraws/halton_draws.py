
from chaospy import create_halton_samples
from MixtureOptDesign.mnl.mnl_utils import *
import numpy as np
from scipy.stats import norm


class HaltonDraws:
    """
    Superclass to generate Halton draws from a multi-normal distribution
    with a specified mean vector and covariance matrix.
    """
    def __init__(self, beta:np.ndarray, cov_matrix:np.ndarray, n:int):
        """
        Initializes the DrawHalton superclass with a mean vector, covariance matrix,
        and number of Halton draws to generate.
        
        Parameters
        ----------
        beta : numpy.ndarray
            Mean vector of the multi-normal distribution.
        cov_matrix : numpy.ndarray
            Covariance matrix of the multi-normal distribution.
        n : int
            Number of Halton draws to generate.
        """
        self.beta = beta
        self.cov_matrix = cov_matrix
        self.n = n
        self.halton_draws = None

    def generate_draws(self):
        """
        Generate Halton draws from a multi-normal distribution with a specified mean vector and covariance matrix
        using create_halton_samples.

        Returns
        -------
        numpy.ndarray
            Halton draws from the multi-normal distribution.
        """
        # Define dimensionality
        dims = self.beta.size
        
        # Generate Halton sequences
        halton_sequences = create_halton_samples(self.n, dim=dims, burnin=12)
        
        # Apply inverse CDF of standard normal distribution
        norm_sequences = norm.ppf(halton_sequences)
        
        # Use predefined covariance matrix to transform standard normal random numbers to correlated normal random numbers
        # perform Cholesky decomposition on the covariance matrix
        correlated_norm_sequences = np.dot(np.linalg.cholesky(self.cov_matrix), norm_sequences)
        
        # Add mean vector to obtain final Halton draws
        self.halton_draws = correlated_norm_sequences.T + self.beta
        
        return self.halton_draws

    def get_average_i_optimality_mnl(self, design, order):
        """
    Calculates the average I-optimality criterion for a multinomial logit model design
    over a set of Halton draws.

    Parameters
    ----------
    design : numpy.ndarray
        The design cube of shape (q, J, S), where q is the number of ingredients,
        J is the number of alternatives, and S is the number of choice sets.
    order : int
        The maximum order of interaction effects to include in the model.

    Returns
    -------
    float
        The average I-optimality criterion value over the set of Halton draws.
    """
        if self.halton_draws is None:
            self.halton_draws = self.generate_draws()
        
        # Calculate the I-optimality criterion for each Halton draw
        i_opt_array = np.zeros(self.n)
        for i in range(self.n):
            beta = self.halton_draws[i]
            i_opt_array[i] = get_i_optimality_mnl(design, order, beta)
        
        # Calculate the average I-optimality criterion over all Halton draws
        i_opt_avg = np.mean(i_opt_array)
        
        return i_opt_avg






