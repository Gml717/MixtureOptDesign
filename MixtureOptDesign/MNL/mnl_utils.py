""""
look at utils.pyx
"""



import numpy as np
from typing import Tuple
import numpy as np
import itertools
from scipy.special import factorial
from numpy.linalg import det



def get_random_initial_design_mnl(n_ingredients: int, n_alternatives: int, n_choice_sets: int, seed: int = None) -> np.ndarray:
    """
    Generate a random initial design for a multinomial logit (MNL) model.

    Parameters:
    -----------
    n_ingredients : int
        The number of ingredients in the MNL model.
    n_alternatives : int
        The number of alternatives in the MNL model.
    n_choice_sets : int
        The number of choice sets in the MNL model.
    seed : int or None, optional
        Seed for the random number generator. If None, a new seed will be used.

    Returns:
    --------
    design : numpy.ndarray, shape (n_ingredients, n_alternatives, n_choice_sets)
        A 3-dimensional array of random values that can be used as an initial design for the MNL model.

    Notes:
    ------
    The values in the resulting array are normalized so that each alternative in each choice set sums to 1.0.

    Examples:
    ---------
    >>> np.random.seed(0)
    >>> get_random_initial_design_mnl(2, 3, 2)
    array([[[0.36995516, 0.46260563, 0.35426968],
            [0.13917321, 0.29266522, 0.50959886],
            [0.49087163, 0.24472915, 0.13613147]],

           [[0.63004484, 0.53739437, 0.64573032],
            [0.86082679, 0.70733478, 0.49040114],
            [0.50912837, 0.75527085, 0.86386853]]])

    """
    if seed is not None:
        np.random.seed(seed)
    random_values = np.random.rand(n_ingredients, n_alternatives, n_choice_sets)
    design = random_values / np.sum(random_values, axis=0)
    return design


def get_choice_probabilities_mnl(design:np.ndarray, beta:np.ndarray, order:int) -> np.ndarray:
    """
    Compute the choice probabilities for a multinomial logit (MNL) model.
    
    Parameters
    ----------
    design : ndarray of shape (q, J, S)
        The design cube where q is the number of ingredients, J is the number of alternatives, and S is the number of choice sets.
        
    beta : ndarray of shape (p,)
        The vector of beta coefficients, where p is the number of parameters in the model.
    order : int
        The maximum order of interactions to include in the model. Must be 1,2 or 3.
    
    Returns
    -------
    P : ndarray of shape (J, S)
        The choice probabilities of the MNL model, where J is the number of alternatives and S is the number of choice sets.
    """
    beta_star, beta_2FI, beta_3FI = get_beta_coefficients(beta, order, design.shape[0])
    U = get_utilities(design, beta_star, beta_2FI, beta_3FI, order)
    P = get_choice_probabilities(U)
    
    
    return P


def get_parameters(q:int, order:int) -> Tuple[int, int, int]:
    """
    Calculate the total number of parameters needed for a given order of interactions in a MNL model.

    Parameters
    ----------
    q : int
        The number of mixture ingredients.
    order : int
        The maximum order of interactions to include in the model. Must be 1, 2 or 3.


    Returns
    -------
   Tuple[int, int, int]
        A tuple containing the number of parameters for the linear, quadratic, and cubic effects.

    """
    
    if order not in [1, 2, 3]:
        raise ValueError("Order must be 1, 2, or 3")
    
    
    p1 = q - 1
    p2 = q * (q - 1)//2
    p3 = q * (q - 1) * (q - 2)//6


    if order == 1:
        return (p1, 0, 0)
    elif order == 2:
        return (p1, p2, 0)
    else:
        return (p1, p2, p3)


def get_beta_coefficients(beta:np.ndarray, q:int, order:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gets the beta coefficients for the different terms in the MNL model.

    Parameters
    ----------
    beta : numpy.ndarray of shape (p,)
        A 1-dimensional array of p numbers of beta coefficients for the model.
    q : int
        The number of ingredients.
    order : int
        The maximum order of interactions to include in the model. Must be 1, 2 or 3.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        A tuple of three numpy.ndarray objects containing the beta coefficients for the linear, quadratic, and cubic effects.

    """
    
    
    p1, p2, p3 = get_parameters(q, order)


    if beta.size != p1 + p2 + p3:
        raise ValueError("Number of beta coefficients does not match the number of parameters for the given order and number of ingredients.")


    beta_star = beta[:p1] 
    beta_2FI = beta[p1:p1+p2] if order >= 2 else np.empty(0)
    beta_3FI = beta[p1 + p2:p1 + p2 +p3] if order == 3 else np.empty(0)


    return beta_star, beta_2FI, beta_3FI


def get_choice_probabilities(U:np.ndarray) -> np.ndarray:
    """
    Calculate choice probabilities from utilities using the MNL model.

    Parameters
    ----------
    U : numpy.ndarray
        2D array of size (J, S) representing the utilities of each alternative for each decision.

    Returns
    -------
    P : numpy.ndarray
        2D array of size (J, S) representing the choice probabilities of each alternative for each decision.
    """
    expU = np.exp(U)
    P = expU / np.sum(expU, axis=0)
    return P


def multiply_arrays(*arg: np.ndarray) -> np.ndarray:
    """
    Multiply multiple numpy arrays element-wise.

    Parameters
    ----------
    *arg : np.ndarray
        Variable number of numpy arrays to multiply.

    Returns
    -------
    np.ndarray
        Numpy array that is the element-wise product of all the input arrays.

    Raises
    ------
    ValueError
        If no input arrays are provided.

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> c = np.array([7, 8, 9])
    >>> multiply_arrays(a, b, c)
    array([ 28,  80, 162])
    """
    if not arg:
        raise ValueError("No input arrays provided.")
    
    result = 1
    
    for i in range(0, len(arg)):
        
        result *= arg[i]
        
    return result

    
def interaction_terms(arr: np.ndarray, interaction: int) -> np.ndarray:
    """
    Compute element-wise multiplication of all pair of combination of axes in a numpy array.

    Parameters:
    -----------
    arr : np.ndarray
        The input array.
    interaction : int
        The number of axes to multiply together. 

    Returns:
    --------
    np.ndarray
        A new array that corresponds to the element-wise multiplication
        of all pair of combination of axes.
    """
    
    if not isinstance(interaction,int): 
         raise TypeError("non-integer interaction")
    elif interaction < 0:
        raise ValueError("interaction is zero or negative")
    elif arr.size == 0:
        raise ValueError("empty array")
    
    arr1 = arr.copy()
    elements = list(range(arr.shape[0]))
    
    pairs = list(itertools.combinations(elements, interaction))
    
    axis_results = [multiply_arrays(*[arr1[i] for i in axes]) for axes in pairs]
    
    return np.stack(axis_results, axis=0)


def get_utilities(design:np.ndarray, beta_star:np.ndarray, beta_2FI:np.ndarray, beta_3FI:np.ndarray, order:int) -> np.ndarray:
    """
    Calculates the utilities for each alternative and choice set in the design cube for MNL model.

    Parameters
    ----------
    design : ndarray of shape (q, J, S)
        The design cube where q is the number of ingredients, J is the number of alternatives, and S is the number of
        choice sets.
    beta_star : ndarray of shape (q-1,)
        The coefficients for the linear term in the MNL model.
    beta_2FI : ndarray of shape (q * (q - 1)//2, )
        The coefficients for the two-factor interaction terms in the MNL model.
    beta_3FI : ndarray of shape (q * (q - 1) * (q - 2)//6,)
        The coefficients for the three-factor interaction terms in the MNL model.
    order : int
        The maximum order of interactions to include in the model. Must be 1, 2 or 3.

    Returns
    -------
    numpy.ndarray
         Array of shape (J, S) containing the utility (matrix) of alternative j in choice set s

    """
    try:
        q, J, S = design.shape
    except ValueError:
        raise ValueError("Design matrix must be a 3-dimensional numpy array with dimensions (q, J, S).")
    
    # Calculate utilities
    

    # Linear term  
    # # ORDER-1 X_JS
    
    x_js = design[:-1]
    U_js_term1 = np.sum(beta_star.reshape(beta_star.size,1,1) * x_js, axis=0)
    U = U_js_term1
    

    # Quadratic term
    # ORDER-2 X_IJS*X_KJS - 2FI terms
    if order >= 2:
        x_js_2 = interaction_terms(design,2)
        U_js_term2= np.sum(beta_2FI.reshape(beta_2FI.size,1,1)*x_js_2, axis=0)
        U += U_js_term2

        # Cubic term
        # 0RDER-3 X_IJS*X_KJS*XLJS - 3FI terms
        if order == 3:
            x_js_3 = interaction_terms(design,3)
            U_js_term3= np.sum(beta_3FI.reshape(beta_3FI.size,1,1)*x_js_3, axis=0)
            U += U_js_term3

    return U


def get_model_matrix(design: np.ndarray, order: int) -> np.ndarray:
    """
    Constructs the model matrix for a multinomial logit(MNL) model.

    Parameters
    ----------
    design : numpy.ndarray
        The design cube of shape (p, J, S), where p is the number of parameters in the model,
        J is the number of alternatives, and S is the number of choice sets.
    order : int
        The maximum order of interaction terms to include in the model matrix. 
        Must be 1, 2, or 3.

    Returns
    -------
    numpy.ndarray
        The model cube of shape (p, J, S), where p is the number of parameters
        in the model.

    Raises
    ------
    ValueError
        If order is not 1, 2, or 3.

    """
    if order not in [1, 2, 3]:
        raise ValueError("Order must be 1, 2, or 3")
    
    q, J, S = design.shape
    p1, p2, p3 = get_parameters(q, order)
    model_array = np.zeros((p1 + p2 + p3, J, S))

    model_array[0:p1,:,:] = design[0:p1, :, :]

    if order >= 2:
        second_order = interaction_terms(design, 2)
        model_array[p1:p1 + p2,:,:] = second_order

    if order == 3:
        third_order = interaction_terms(design, 3)
        model_array[p1 + p2:p1 + p2 + p3,:,:] = third_order

    return model_array


def get_information_matrix_mnl(design: np.ndarray, order: int, beta:np.ndarray)->np.ndarray:
    """
    Get the information matrix for design and parameter beta.
    The function returns the sum of the information matrices of the S choice sets.

    Parameters
    ----------
    design : np.ndarray
        The design cube of shape (q, J, S), where q is the number of ingredients,
        J is the number of alternatives, and S is the number of choice sets.
    order : int 
        The polynomial order of the design cube.
    beta : np.ndarray 
        The parameter vector of shape (M,).

    Returns:
    np.ndarray: The information matrix of shape (M, M).
    
    """
    
    
    Xs = get_model_matrix(design,order)
    
    param, J, S = Xs.shape
    
    P = get_choice_probabilities_mnl(design,beta,order)
    
    
    
    information_matrix = 0
    for s in range(S):
        p_s = P[:, s]
        I_s = np.dot(Xs[:param, :, s], np.dot(np.diag(p_s) - np.outer(p_s, p_s.T), Xs[:param, :, s].T))
        information_matrix += I_s
    
    
    
    
    return information_matrix
    
def get_moment_matrix(q:int, order:int) -> np.ndarray:
    """
    Computes the moment matrix for a multinomial logit (MNL) model of order 1, 2 or 3.
    
    Parameters:
    -----------
    q : int
        The number of mixture ingredients.
    order : int
        The order of the MNL model (1, 2, or 3).
        
    Returns:
    --------
    np.ndarray
        The moment matrix of size (parameters, parameters), where parameters is the number
        of parameters in the MNL model.
    
    Raises:
    -------
    ValueError
        If order is not 1, 2 or 3.
    """

    if order not in [1, 2, 3]:
        raise ValueError("Order must be 1, 2, or 3")
    

    p1,p2,p3 = get_parameters(q,order)
    parameters = p1 + p2 + p3
    
    
    auxiliary_matrix = np.zeros((parameters, q))
    auxiliary_matrix[:q-1, :q-1] = np.eye(q-1)
    counter = q - 2
    
    if order >= 2:
        for i in range(q-1):
            for j in range(i+1, q):
                counter += 1
                auxiliary_matrix[counter, i] = 1
                auxiliary_matrix[counter, j] = 1
    
                
    if order >= 3:
        for i in range(q-2):
            for j in range(i+1, q-1):
                for k in range(j+1, q):
                    counter += 1
                    auxiliary_matrix[counter, i] = 1
                    auxiliary_matrix[counter, j] = 1
                    auxiliary_matrix[counter, k] = 1
    
                     
    W = np.zeros((parameters, parameters))
    for i in range(parameters):
        aux_sum = auxiliary_matrix[i] + auxiliary_matrix
        num = np.product(factorial(aux_sum),axis =1)
        denom = factorial(q -1 + np.sum(aux_sum,axis=1))
        W[i,:] = num/denom   
        
    return W
 

def get_i_optimality_mnl(design: np.ndarray, order: int, beta: np.ndarray) -> float:
    
    """
    Calculates the I-optimality criterion for a multinomial logit model design.

    Parameters
    ----------
    design : numpy.ndarray
        The design cube of shape (q, J, S), where q is the number of ingredients,
        J is the number of alternatives, and S is the number of choice sets.
    order : int
        The maximum order of interaction effects to include in the model.
    beta : numpy.ndarray
        The parameter vector of shape (p, ) for the MNL model.

    Returns
    -------
    i_opt : float
        The I-optimality criterion value for the MNL design.
    """

    q = design.shape[0]
    information_matrix = get_information_matrix_mnl(design, order, beta)
    moments_matrix = get_moment_matrix(q, order)
    
    #The sum of all  diagonal elements of the product of the inverse information producct and moment matrix. This is the same as computing the trace of the matrix product for these two matrices.
    i_opt = np.trace(np.linalg.solve(information_matrix, moments_matrix))
    return i_opt  


def generate_beta_params(num_params:int, q:int) -> np.ndarray:
    """
    Generate a set of beta parameters from a multinormal distribution.

    Parameters
    ----------
    num_params : int
        The number of parameters in the multinormal distribution.
    q : int
        The number of mixture ingredients.
    
    Returns
    -------
    numpy.ndarray
        A vector of beta parameters from the multinormal distribution,
        with the `q`-th parameter removed and all previous parameters subtracted
        by thhis value.

    Raises
    ------
    ValueError
        If `q` is less than 1 or greater than `num_params`.

    Examples
    --------
    >>> generate_beta_params(5, 3)
    array([ 0.0594564 ,  0.2054975 , -0.07127753, -0.02105723])
    
    """
    remove_idx = q - 1
    if remove_idx < 0 or remove_idx >= num_params:
        raise ValueError("q must be between 1 and the number of parameters")
    mean = np.zeros(num_params)
    cov = np.identity(num_params)
    beta_params = np.random.multivariate_normal(mean, cov)
    
    for i in range(remove_idx):
        beta_params[i] -= beta_params[remove_idx]
    return np.concatenate([beta_params[:remove_idx], beta_params[remove_idx+1:]])


def compute_cox_direction(q: np.ndarray, index: int, n_points: int = 30) -> np.ndarray:
    """
    Computes the Cox direction for a given index of q and number of points.

    Parameters
    ----------
    q : np.ndarray
        A 1-dimensional ndarray of the mixture proportions. Must sum up to one
    index : int
        The index of the proportion for which the Cox direction is calculated.
    n_points : int, optional
        The number of points to generate in the sequence, by default 30.

    Returns
    -------
    np.ndarray
        A 2-dimensional ndarray of shape (n_points, q.size) representing the Cox direction. Dimension 2 must sum up to one

    """
    cox_direction = np.empty((n_points, q.size))
    prop_sequence = np.linspace(0, 1, n_points)
    qi = q[index]
    delta = prop_sequence - qi
    k_mask = np.arange(q.size) != index
    k_values = q[k_mask]
    if np.isclose(qi, 1):
        cox_direction[:, k_mask] = (1 - prop_sequence[:, np.newaxis]) / (q.size - 1)
    else:
        cox_direction[:, k_mask] = (1 - delta[:, np.newaxis] / (1 - qi)) * k_values
    cox_direction[:, index] = prop_sequence
    return cox_direction



def get_d_optimality(design:np.ndarray, order:int, beta:np.ndarray)-> float:
    info_matrix = get_information_matrix_mnl(design,order,beta)
    d_value = np.log(det(info_matrix)**(-1/beta.size))
    return d_value



def transform_varcov_matrix(id_matrix: np.ndarray, q: int, k = 1) -> np.ndarray:
    """
    Transform a variance-covariance matrix by adding a constant value to the 
    diagonal of a subset of rows and columns.

    Parameters
    ----------
    id_matrix : numpy.ndarray
        The identity matrix to be transformed.
    q : int
        The number of ingredients to add a constant value to the diagonal.
    k :  [optional, default 1] int 
        positive scalar that controls the level of uncertainty

    Returns
    -------
    numpy.ndarray
        The transformed variance-covariance matrix.
    """
    num_param = id_matrix.shape[0]
    Sigma_prime = np.zeros((num_param-1, num_param-1))

    Sigma_prime[0:q-1, 0:q-1] = id_matrix[0:q-1, 0:q-1] + id_matrix[q-1, q-1]
    Sigma_prime[q-1:, q-1:] = id_matrix[q:, q:]
    
    return Sigma_prime *k


def get_i_optimality_bayesian(design: np.ndarray, order: int, beta: np.ndarray) -> float:
    
    #apply a function to each row. 
    i_opt_array = np.apply_along_axis(lambda beta_array: get_i_optimality_mnl(design, order, beta_array), 1, beta)
    return np.mean(i_opt_array)