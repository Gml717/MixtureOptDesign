import numpy as np
cimport numpy as np
import warnings


from libc.math cimport log


cpdef double factorial(int n):
    cdef double result = 1
    cdef int i
    
    for i in range(2, n+1):
        result *= i
        
    return result





cpdef  factorial_2d(np.ndarray[np.double_t, ndim=2] arr):
    cdef np.ndarray[np.int64_t, ndim=2] result
    result = np.vectorize(factorial, otypes=[np.int64])(arr)
    return result

cpdef  factorial_1d(np.ndarray[np.double_t, ndim=1] arr):
    cdef np.ndarray[np.int64_t, ndim=1] result
    result = np.vectorize(factorial, otypes=[np.int64])(arr)
    return result


cpdef  get_random_initial_design_mnl(int n_ingredients, int n_alternatives, int n_choice_sets, int seed):
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
    cdef np.ndarray[np.float64_t, ndim=3] random_values
    cdef np.ndarray[np.float64_t, ndim=3] design
    
    
    np.random.seed(seed)
    random_values = np.random.rand(n_ingredients, n_alternatives, n_choice_sets)
    design = random_values / np.sum(random_values, axis=0)
    return design





cpdef  get_choice_probabilities_mnl(np.ndarray[np.double_t, ndim=3] design, np.ndarray[np.double_t, ndim=1] beta, int order):
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
    cdef np.ndarray[np.double_t, ndim=1] beta_star
    cdef np.ndarray[np.double_t, ndim=1] beta_2FI
    cdef np.ndarray[np.double_t, ndim=1] beta_3FI
    beta_star, beta_2FI, beta_3FI = get_beta_coefficients(beta, order, design.shape[0])
    
    cdef np.ndarray[np.double_t, ndim=2] U
    U = get_utilities(design, beta_star, beta_2FI, beta_3FI, order)
    
    cdef np.ndarray[np.double_t, ndim=2] P
    P = get_choice_probabilities(U)
    
    return P


cpdef  get_choice_probabilities(np.ndarray[np.double_t, ndim=2] U):
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
    # subtracting the maximum value to avoid numerical overflow
    # - np.max(U,axis=1).reshape(U.shape[0],1)
    cdef np.ndarray[np.double_t, ndim=2] expU
    expU = np.exp(U - np.max(U,axis=0) ) 
    cdef np.ndarray[np.double_t, ndim=2] P
    P = expU / np.sum(expU, axis=0)
    return P



cpdef get_utilities(np.ndarray[np.double_t, ndim=3] design, np.ndarray[np.double_t, ndim=1] beta_star,np.ndarray[np.double_t, ndim=1] beta_2FI, np.ndarray[np.double_t, ndim=1] beta_3FI,int order):
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
    cdef int q, J, S, i, j, s, k, l
    try:
        q, J, S = np.shape(design)
    except ValueError:
        raise ValueError("Design matrix must be a 3-dimensional numpy array with dimensions (q, J, S).")
    
    cdef np.ndarray[np.double_t, ndim=3] x_js, x_js_2, x_js_3
    cdef np.ndarray[np.double_t, ndim=2] U_js_term1, U_js_term2, U_js_term3
    cdef np.ndarray[np.double_t, ndim=2] U
    
    
    # Linear term
    x_js = design[:-1]
    U_js_term1 = np.sum(beta_star.reshape(beta_star.size,1,1) * x_js, axis=0)
    U = U_js_term1
    
    # Quadratic term
    if order >= 2:
        x_js_2 = interaction_terms(design,2)
        U_js_term2 = np.sum(beta_2FI.reshape(beta_2FI.size,1,1) * x_js_2, axis=0)
        U += U_js_term2
        
    # Cubic term
    if order == 3:
        x_js_3 = interaction_terms(design,3)
        U_js_term3 = np.sum(beta_3FI.reshape(beta_3FI.size,1,1) * x_js_3, axis=0)
        U+= U_js_term3

    return U




cpdef   get_beta_coefficients(np.ndarray[np.double_t, ndim=1] beta, int q, int order):
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
    cdef int p1, p2, p3
    p1, p2, p3 = get_parameters(q, order)

    if beta.size != p1 + p2 + p3:
        raise ValueError("Number of beta coefficients does not match the number of parameters for the given order and number of ingredients.")

    cdef np.ndarray[np.double_t, ndim=1] beta_star, beta_2FI, beta_3FI
    beta_star = beta[:p1] 
    beta_2FI = beta[p1:p1+p2] if order >= 2 else np.empty(0)
    beta_3FI = beta[p1 + p2:p1 + p2 +p3] if order == 3 else np.empty(0)

    return beta_star, beta_2FI, beta_3FI



cpdef tuple[int] get_parameters(int q, int order):
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
    
    cdef int p1, p2, p3
    p1 = q - 1
    p2 = q * (q - 1)//2
    p3 = q * (q - 1) * (q - 2)//6

    if order == 1:
        return (p1, 0, 0)
    elif order == 2:
        return (p1, p2, 0)
    else:
        return (p1, p2, p3)

import numpy as np
cimport numpy as np

cpdef multiply_arrays(arg1=None, arg2=None, arg3=None):
    """
    Multiply multiple numpy arrays element-wise.

    Parameters
    ----------
    arg1 : np.ndarray
        First numpy array to multiply.
    arg2 : np.ndarray, optional
        Second numpy array to multiply. Default is None.
    arg3 : np.ndarray, optional
        Third numpy array to multiply. Default is None.

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

    >>> d = np.array([10, 11, 12])
    >>> multiply_arrays(a, b, d)
    array([ 280,  880, 2160])

    >>> multiply_arrays(a)
    array([1, 2, 3])
    """
    if arg1 is None and arg2 is None and arg3 is None:
        raise ValueError("All three arguments cannot be None")

    if arg2 is None and arg3 is None:
        return arg1
    elif arg3 is None:
        return arg1 * arg2
    else:
        return arg1 * arg2 * arg3






import numpy as np
cimport numpy as np
import itertools

cpdef np.ndarray interaction_terms(np.ndarray arr, int interaction):
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

    cdef np.ndarray arr1 = arr.copy()
    cdef list elements = list(range(arr.shape[0]))
    
    cdef list pairs = list(itertools.combinations(elements, interaction))
    
    cdef list axis_results = [multiply_arrays(*[arr1[i] for i in axes]) for axes in pairs]
    
    return np.stack(axis_results, axis=0)


cpdef  get_model_matrix(np.ndarray[np.float64_t, ndim=3] design, int order):
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
    cdef int q,J,S
    
    cdef int p1, p2, p3
    q,J,S = np.shape(design)

    if order not in [1, 2, 3]:
        raise ValueError("Order must be 1, 2, or 3")

    p1, p2, p3 = get_parameters(q, order)
        
    cdef np.ndarray[np.float64_t, ndim=3] model_array = np.zeros((p1 + p2 + p3, J, S), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] second_order
    cdef np.ndarray[np.float64_t, ndim=3] third_order
    
    model_array[0:p1,:,:] = design[0:p1, :, :]

    if order >= 2:
        second_order = interaction_terms(design, 2)
        model_array[p1:p1 + p2,:,:] = second_order

    if order == 3:
        third_order = interaction_terms(design, 3)
        model_array[p1 + p2:p1 + p2 + p3,:,:] = third_order

    return model_array


import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound


# Define the get_information_matrix_mnl function in Cython

cpdef  get_information_matrix_mnl(np.ndarray[np.float64_t, ndim=3] design, int order, np.ndarray[np.float64_t, ndim=1] beta):
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

    cdef np.ndarray[np.float64_t, ndim=3] Xs = get_model_matrix(design, order)

    cdef int param = Xs.shape[0]
    cdef int J = Xs.shape[1]
    cdef int S = Xs.shape[2]

    cdef np.ndarray[np.float64_t, ndim=2] P = get_choice_probabilities_mnl(design, beta, order)

    cdef np.ndarray[np.float64_t, ndim=2] I_s, information_matrix = np.zeros((param,param),dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] p_s
    cdef int s

    for s in range(S):
        p_s = P[:, s]
        I_s = np.dot(Xs[:param, :, s], np.dot(np.diag(p_s) - np.outer(p_s, p_s.T), Xs[:param, :, s].T))
        information_matrix += I_s
    
    return information_matrix



cpdef get_moment_matrix(int q, int order):
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

    cdef int p1, p2, p3, parameters, counter, i, j, k
    p1, p2, p3 = get_parameters(q, order)
    parameters = p1 + p2 + p3
    cdef np.ndarray[np.double_t, ndim=2] auxiliary_matrix = np.zeros((parameters, q))
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

    cdef np.ndarray[np.double_t, ndim=2] W = np.zeros((parameters, parameters))
    cdef np.ndarray[np.double_t, ndim=2] aux_sum
    cdef np.ndarray[np.double_t, ndim=1] puk
    cdef np.ndarray[np.int64_t, ndim=1] denom
    for i in range(parameters):
        aux_sum = auxiliary_matrix[i] + auxiliary_matrix
        num = np.product(factorial_2d(aux_sum),axis =1)
        puk = q -1 + np.sum(aux_sum,axis=1)
        denom = factorial_1d(puk)
        W[i,:] = num/denom  

    return W




cpdef float get_i_optimality_mnl(np.ndarray[np.float64_t, ndim=3] design, int order, np.ndarray[np.float64_t, ndim=1] beta):
    
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

    cdef int q = design.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] information_matrix = get_information_matrix_mnl(design, order, beta)
    cdef np.ndarray[np.float64_t, ndim=2] moments_matrix = get_moment_matrix(q, order)
    cdef float i_opt 
    cdef float default_i_opt = 100

    try:
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")


            i_opt = np.log((np.trace(np.linalg.solve(information_matrix, moments_matrix))))
                # Iterate over the collected warnings and print the message
            for warning in w:
                if "invalid value encountered in log" in str(warning.message):
                    raise ValueError("Error: invalid value encountered in log")


    except ValueError as e:
        return default_i_opt 

    except np.linalg.LinAlgError:
        return default_i_opt 

    

    return i_opt


import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, srand, RAND_MAX

cpdef  generate_beta_params(int num_params, int q):
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
    cdef int remove_idx = q - 1
    if remove_idx < 0 or remove_idx >= num_params:
        raise ValueError("q must be between 1 and the number of parameters")
    cdef np.ndarray[double, ndim=1] mean = np.zeros(num_params)
    cdef np.ndarray[double, ndim=2] cov = np.identity(num_params)
    cdef np.ndarray[double, ndim=1] beta_params =  np.random.multivariate_normal(mean, cov)
    
    cdef int i

    for i in range(remove_idx):
        beta_params[i] -= beta_params[remove_idx]
    return np.concatenate([beta_params[:remove_idx], beta_params[remove_idx+1:]])


import numpy as np
cimport numpy as np

cpdef  compute_cox_direction(np.ndarray[np.float64_t, ndim=1] q, int index, int n_points=30):
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
    cdef np.ndarray[np.float64_t, ndim=2] cox_direction = np.empty((n_points, q.size), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] prop_sequence = np.linspace(0, 1, n_points)
    cdef np.float64_t qi = q[index]
    cdef np.ndarray[np.float64_t, ndim=1] delta = prop_sequence - qi

    cdef np.ndarray[np.npy_bool, ndim=1] k_mask = np.arange(q.size) != index
    cdef np.ndarray[np.float64_t, ndim=1] k_values = q[k_mask]
    
    if np.isclose(qi, 1):
        cox_direction[:, k_mask] = (1 - prop_sequence[:, np.newaxis]) / (q.size - 1)
    else:
        cox_direction[:, k_mask] = (1 - delta[:, np.newaxis] / (1 - qi)) * k_values
    cox_direction[:, index] = prop_sequence
    return cox_direction


import numpy as np
cimport numpy as np
from numpy.linalg import det

cpdef double get_d_optimality(np.ndarray[np.double_t, ndim=3] design, int order, np.ndarray[np.double_t, ndim=1] beta):
    cdef np.ndarray[np.double_t, ndim=2] info_matrix = get_information_matrix_mnl(design, order, beta)
    cdef double d_value = np.log(det(info_matrix)**(-1/beta.size))
    return d_value

cpdef  transform_varcov_matrix(np.ndarray[np.double_t, ndim=2] id_matrix, int q, double k=1):
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
    cdef int num_param = id_matrix.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] Sigma_prime = np.zeros((num_param-1, num_param-1))

    Sigma_prime[0:q-1, 0:q-1] = id_matrix[0:q-1, 0:q-1] + id_matrix[q-1, q-1]
    Sigma_prime[q-1:, q-1:] = id_matrix[q:, q:]
    
    return Sigma_prime * k





cpdef double get_i_optimality_bayesian(np.ndarray[np.double_t, ndim=3] design, int order, np.ndarray[np.double_t, ndim=2] beta):
    cdef int n = beta.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] i_opt_array = np.zeros(n, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] beta_row
    cdef double i_opt_avg
    cdef int i
    cdef float default_i_opt = 100
    
    for i in range(n):
        beta_row = beta[i]
        i_opt_array[i] = get_i_optimality_mnl(design, order, beta_row)
        if i_opt_array[i] == default_i_opt:
            return default_i_opt
        
    
    i_opt_avg = np.mean(i_opt_array)
    return i_opt_avg
