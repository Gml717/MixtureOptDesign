import numpy as np
from itertools import product,combinations
import numpy as np

from MixtureOptDesign.vns.vns_cython import unique_rows

def generate_simplex_lattice_design(q:int, m:int)-> np.ndarray:
    """
    Generate a simplex lattice design for q components and m levels.

    Parameters:
    -----------
    q : int
        The number of components in the design.
    m : int
        The number of equally spaced levels for each component.

    Returns:
    --------
    design : numpy.ndarray, shape (n_points, q)
        A 2-dimensional array of design points with n_points rows and q columns.

    Notes:
    ------
    The values in the resulting array are the proportions of each component and are equally spaced between 0 and 1.

    Examples:
    ---------
    >>> generate_simplex_lattice_design(3, 3)
    array([[0.        , 0.        , 1.        ],
       [0.        , 0.33333333, 0.66666667],
       [0.        , 0.66666667, 0.33333333],
       [0.        , 1.        , 0.        ],
       [0.33333333, 0.        , 0.66666667],
       [0.33333333, 0.33333333, 0.33333333],
       [0.33333333, 0.66666667, 0.        ],
       [0.66666667, 0.        , 0.33333333],
       [0.66666667, 0.33333333, 0.        ],
       [1.        , 0.        , 0.        ]])
    """
    if q <= 0 or m <= 0:
        raise ValueError("values should be positive integer")
    
    levels = np.linspace(0, 1, m+1)
    points = np.array(list(product(levels, repeat=q)))
    point_sums = np.sum(points, axis=1)
    design = points[np.isclose(point_sums, 1)]
    return design


def generate_initial_design(lattice_design, j=2, s=16,k=10):
    """
    Generate an initial design by selecting k points randomly from a given lattice design, allocate those points randomely  in a design of shape (q,j,s) with s choice sets with j alternatives based on all possible combination of points k of size.
    
    Parameters
    ----------
    lattice_design : numpy array
        A 2D numpy array of shape (n, q), representing the lattice design where n is the number
        of points in the canditate sets and q is the number of ingredients.
    j : int, optional
        The number of alternatives in each choice set. Default is 2.
    s : int, optional
        The number of choice sets to create. Default is 16.
    k : int, optional
        The number of of distinct point inthe design
    
    Returns
    -------
    numpy array
        intial design with shape (q,j,s)
   
    numpy array
        lattice points, k random points from lattice design that are included in the initial design with shape (k, p)
        
    numpy array
        other points not included in the initial design  with shape (n-k, p), where n is the total number of points in the lattice design and k is the number of lattice points included in the construction of the initial design.
        
    Raises
    ------
    ValueError
        If j is less than 2 or greater than k.
    
    Notes
    -----
    This function uses the numpy.random.choice function to select k points randomly from the given
    lattice design, and the itertools.combinations function to generate all possible combinations
    of j points from the k points. Then, a random subset of s choice sets are selected from the
    generated combinations.
    """
    
    # get the total number of points in the lattice design
    n = lattice_design.shape[0]
    
    # ensure j is between 2 and k
    if j < 2 or j > k:
        raise ValueError("j must be between 2 and k")
    
    
    # initialize number of unique rows to 0
    unique_rows_count = 0
    
    # keep generating an initial design until it has k distinct points
    while unique_rows_count != k:

        # generate k random indices
        lattice_indices = np.random.choice(n, size=k, replace=False)

        # get k random points
        lattice_points = lattice_design[lattice_indices]

        # get (n-k) other points
        other_points = np.delete(lattice_design, lattice_indices, axis=0)
        
        ## Allocate points randomely  in a design with j alternatives and s choice sets

        # generate all possible combinations of lattice(k) points of size j
        choice_set = np.array(list(combinations(lattice_points, j)))

        # choose s random combinations from the list of all possible combinations
        choice_indices = np.random.choice(choice_set.shape[0], size=s, replace=False)

        # create the initial design
        initial_design = choice_set[choice_indices,:,:]
        
        # reshape initial design 
        initial_design = initial_design.transpose(2, 1, 0)
        # get the number of unique rows in the initial design
        unique_rows_count= unique_rows(initial_design).shape[0]
    
    return initial_design, lattice_points, other_points
