import numpy as np
import pytest

from Tests.utils import check_mnl_design_sum
from MixtureOptDesign.data.csv_util import read_csv_file
from MixtureOptDesign.mnl.utils import *

@pytest.fixture
def arr():
    design = np.array([
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0, 0],
            ],
            [
                [0.2, 0.3, 0.4],
                [0.5, 0.5, 0.2],
                [0.1, 0, 1.0],
            ],
            [
                [0.7, 0.5, 0.3],
                [0.1, 0, 0.2],
                [0.2, 1.0, 0],
            ],
        ])
        
        
    return design

@pytest.fixture
def design():
    design = np.array([[[0.3897873 , 0.6292498 , 0.7009263 , 0.4262842 , 0.24607377,
         0.48413014, 0.2913878 , 0.72010463],
        [0.4982357 , 0.00751996, 0.2505569 , 0.154967  , 0.5415304 ,
         0.3639989 , 0.04235179, 0.3279013 ]],

       [[0.2375383 , 0.1180457 , 0.2000333 , 0.2569634 , 0.74900543,
         0.03588967, 0.3591402 , 0.00090224],
        [0.2549352 , 0.63608026, 0.4460236 , 0.6877322 , 0.2635403 ,
         0.3727396 , 0.44510098, 0.2701244 ]],

       [[0.3726744 , 0.2527044 , 0.0990404 , 0.3167524 , 0.0049208 ,
         0.47998018, 0.349472  , 0.27899313],
        [0.2468291 , 0.35639978, 0.3034196 , 0.1573008 , 0.1949293 ,
         0.2632615 , 0.51254723, 0.4019743 ]]])
        
        
    return design 
      

class TestInitialDesign(object):
    
    # test case to check that the function returns an array with the correct shape
    def test_design_shape(self) -> None:
        n_ingredients = 3
        n_alternatives = 4
        n_choice_sets = 5
        seed=123
        result = get_random_initial_design_mnl(n_ingredients, n_alternatives, n_choice_sets,seed)
        assert result.shape == (n_ingredients, n_alternatives, n_choice_sets), f"Resulting shape is {result.shape}, expected {(n_ingredients, n_alternatives, n_choice_sets)}"

    # test case to check that the function returns an array where each alternative's values sum to 1
    def test_sum(self) -> None :
        n_ingredients = 3
        n_alternatives = 4
        n_choice_sets = 5
        seed=123
        result = get_random_initial_design_mnl(n_ingredients, n_alternatives, n_choice_sets,seed)
        check_mnl_design_sum(result)

    # test case to check that the function returns different results when using different random seeds
    def test_seed(self) -> None:
        n_ingredients = 3
        n_alternatives = 4
        n_choice_sets = 5
        result1 = get_random_initial_design_mnl(n_ingredients, n_alternatives, n_choice_sets, seed=123)
        result2 = get_random_initial_design_mnl(n_ingredients, n_alternatives, n_choice_sets, seed=456)
        assert not np.allclose(result1, result2)

class TestGetParameters(object):
    
    def test_order_1(self)->None:

        # # Test case for order 1 and q=3
        q = 3
        order = 1
        expected_result = (q -1,0 ,0)
        assert get_parameters(q, order) == expected_result
    
    def test_order_2(self)->None:
        # # Test case for order 2 and q=4   
        q = 4
        order = 2
        expected_result = (q-1,q * (q - 1)/2, 0)
        assert get_parameters(q, order) == expected_result
    
    def test_order_3(self)->None:
        # # Test case for order 3 and q=5
        q = 5
        order = 3
        expected_result = (q-1, q * (q - 1)/2, q * (q - 1) * (q - 2)/6)
        assert get_parameters(q, order) == expected_result
    
class TestGetBetaCoefficients(object):
    
    def test_linear(self):
        beta = np.arange(1,4,dtype=np.double)
        q = 4
        order = 1
        expected_output = (np.arange(1,4), np.empty((0,)), np.empty((0,)))
        assert all(np.array_equal(x, y) for x, y in zip(get_beta_coefficients(beta, q, order), expected_output))
        
        
    def test_quadratic(self):
        beta = np.arange(1,10,dtype=np.double)
        q = 4
        order = 2
        expected_output = (np.arange(1,4), np.arange(4,10), np.empty((0,)))
        assert all(np.array_equal(x, y) for x, y in zip(get_beta_coefficients(beta, q, order), expected_output))


    def test_cubic(self):
        beta = np.arange(1,7,dtype=np.double)
        q = 3
        order = 3
        expected_output = (np.arange(1,3), np.arange(3,6), np.arange(6,7))
        # Test for equality
        assert all(np.array_equal(x, y) for x, y in zip(get_beta_coefficients(beta, q, order), expected_output))


    def test_order_error(self):
        beta = np.random.rand(10)
        q = 4
        order = 4
        with pytest.raises(ValueError):
            get_beta_coefficients(beta, q, order)
            
    def test_beta_error(self):
        beta = np.array([1, 2, 3, 4, 5, 6],dtype=np.double)
        q = 4
        order = 2
        with pytest.raises(ValueError):
            get_beta_coefficients(beta, q, order)

class TestMultiplyArrays(object):


    def test_multiply_arrays(self):
    
        a = np.array([1, 2, 3],dtype=np.float64)
        b = np.array([4, 5, 6],dtype=np.float64)
        c = np.array([7, 8, 9],dtype=np.float64)
        expected_result = np.array([28, 80, 162],dtype=np.float64)
        assert np.array_equal(multiply_arrays(a, b, c), expected_result)

    def test_multiply_arrays_empty_input(self):
        with pytest.raises(ValueError):
            multiply_arrays()

        
class TestInteractionTerms(object):
    
    
    def test_2_order_interaction(self,arr)-> None:
        # Test with interaction = 2
        
    
        axis0_axis1 = np.multiply(arr[0], arr[1])
        axis0_axis2 = np.multiply(arr[0], arr[2])
        axis1_axis2 = np.multiply(arr[1], arr[2])
        interaction = np.stack((axis0_axis1, axis0_axis2, axis1_axis2), axis=0)
        assert np.array_equal(interaction,interaction_terms(arr,2))


    def test_3_order_interaction(self,arr)-> None:
        # Test with interaction = 3
        
        
        axis0_axis1_axis2 = arr[0] * arr[1] * arr[2]
        interaction = axis0_axis1_axis2.reshape(1,3,3)
        assert np.array_equal(interaction,interaction_terms(arr,3))


    # def test_interaction_non_int(self,arr):
    #     # Test with invalid input: non-integer interaction
    #     arr = np.random.rand(3, 4)
    #     with pytest.raises(TypeError):
    #         interaction_terms(arr, 2.5)
        
    def test_empty_array(self,arr):
        # Test with invalid input: empty array
        arr = np.array([])
        with pytest.raises(ValueError):
            interaction_terms(arr, 2)
            
class TestModelMatrix(object):
    
    def test_order_1(self,arr):
        # Test order 1 with small design matrix
        
        expected_model = np.array([[[0.1, 0.2, 0.3],
                                    [0.4, 0.5, 0.6],
                                    [0.7, 0. , 0. ]],

                                    [[0.2, 0.3, 0.4],
                                    [0.5, 0.5, 0.2],
                                    [0.1, 0. , 1. ]]])
        
        result_model = get_model_matrix(arr, 1)
        assert np.array_equal(result_model, expected_model)

    def test_order_2(self,arr):
        # Test order 2 with small design matrix
        expected_model = np.array([[[0.1 , 0.2 , 0.3 ],
                                    [0.4 , 0.5 , 0.6 ],
                                    [0.7 , 0.  , 0.  ]],

                                [[0.2 , 0.3 , 0.4 ],
                                    [0.5 , 0.5 , 0.2 ],
                                    [0.1 , 0.  , 1.  ]],

                                [[0.02, 0.06, 0.12],
                                    [0.2 , 0.25, 0.12],
                                    [0.07, 0.  , 0.  ]],

                                [[0.07, 0.1 , 0.09],
                                    [0.04, 0.  , 0.12],
                                    [0.14, 0.  , 0.  ]],

                                [[0.14, 0.15, 0.12],
                                    [0.05, 0.  , 0.04],
                                    [0.02, 0.  , 0.  ]]])
        result_model = get_model_matrix(arr, 2)
        assert np.allclose(result_model,expected_model)
        #np.testing.assert_array_almost_equal(result_model, expected_model)
        

    def test_order_3(self,arr):
        # Test order 3 with small design matrix
        expected_model = np.array([[[0.1  , 0.2  , 0.3  ],
                                    [0.4  , 0.5  , 0.6  ],
                                    [0.7  , 0.   , 0.   ]],

                                [[0.2  , 0.3  , 0.4  ],
                                    [0.5  , 0.5  , 0.2  ],
                                    [0.1  , 0.   , 1.   ]],

                                [[0.02 , 0.06 , 0.12 ],
                                    [0.2  , 0.25 , 0.12 ],
                                    [0.07 , 0.   , 0.   ]],

                                [[0.07 , 0.1  , 0.09 ],
                                    [0.04 , 0.   , 0.12 ],
                                    [0.14 , 0.   , 0.   ]],

                                [[0.14 , 0.15 , 0.12 ],
                                    [0.05 , 0.   , 0.04 ],
                                    [0.02 , 0.   , 0.   ]],

                                [[0.014, 0.03 , 0.036],
                                    [0.02 , 0.   , 0.024],
                                    [0.014, 0.   , 0.   ]]])
        
        
        result_model = get_model_matrix(arr, 3)
        assert np.allclose(result_model, expected_model)
        #np.testing.assert_array_almost_equal

    def test_order_out_of_bounds(self,arr):
        
        with pytest.raises(ValueError):
            get_model_matrix(arr, 4)
        
class TestGetUtilities(object):
    
    def test_order_1(self,design):

            
            beta_star = np.array([1.0, 2.0])
            beta_2FI = np.array([3.0,4.0, 5.0])
            beta_3FI = np.array([6.0])
            order = 1
            
            n_alternatives = 2
            n_choice_sets = 8
                       
            expected_output = np.array([[0.8648639 , 0.8653412 , 1.1009929 , 0.940211  , 1.74408463,
                                            0.55590948, 1.0096682 , 0.72190911],
                                        [1.0081061 , 1.27968048, 1.1426041 , 1.5304314 , 1.068611  ,
                                            1.1094781 , 0.93255375, 0.8681501 ]])
            U = get_utilities(design, beta_star, beta_2FI, beta_3FI, order)
            assert U.shape == (n_alternatives, n_choice_sets), f"Resulting shape is {U.shape}, expected {(n_alternatives, n_choice_sets)}"
            
            
            assert np.allclose(U, expected_output)


    def test_order_2(self,design):
        
            
            beta_star = np.array([1.0, 2.0])
            beta_2FI = np.array([3.0,4.0, 5.0])
            beta_3FI = np.array([6.0])
            order = 2
            
            n_alternatives = 2
            n_choice_sets = 8
            
            expected_output = np.array([[2.16630935, 1.87339201, 1.89835568, 2.21590435, 2.32028845,
                                            1.62365843, 2.35849015, 1.5287338 ],
                                        [2.19570296, 2.43824515, 2.45862398, 2.48856865, 2.17585545,
                                            2.39045603, 2.21661176, 2.20401944]])
                                                
            U = get_utilities(design, beta_star, beta_2FI, beta_3FI, order)
            assert U.shape == (n_alternatives, n_choice_sets), f"Resulting shape is {U.shape}, expected {(n_alternatives, n_choice_sets)}"
            
            assert np.allclose(U, expected_output)
            
            
    def test_order_3(self,design):
        
            
            beta_star = np.array([1.0, 2.0])
            beta_2FI = np.array([3.0,4.0, 5.0])
            beta_3FI = np.array([6.0])
            order = 3
            
            n_alternatives = 2
            n_choice_sets = 8
            
            expected_output = np.array([[2.37334357, 1.98601766, 1.98167357, 2.42408563, 2.32573018,        1.67369715, 2.57792167, 1.52982139],      [2.38381312, 2.44847374, 2.66207463, 2.58915539, 2.34277156,        2.6047669 , 2.27458339, 2.41764661]])
            
            U = get_utilities(design, beta_star, beta_2FI, beta_3FI, order)
            assert U.shape == (n_alternatives, n_choice_sets), f"Resulting shape is {U.shape}, expected {(n_alternatives, n_choice_sets)}"
            
            assert np.allclose(U, expected_output)
            
class TestGenerateBetaParam(object):
    def test_1D_array(self):
        
        # Test that the function returns a 1-D numpy array
        beta = generate_beta_params(5, 3)
        assert isinstance(beta, np.ndarray)
        assert beta.ndim == 1
        
    
    def test_length_array(self):
        
        # Test that the length of the returned array is correct
        beta = generate_beta_params(5, 3)
        assert len(beta) == 4
        
    
    def test_value_error(self):
        
        # Test that the function raises a ValueError if q is less than 1 or greater than num_params
        with pytest.raises(ValueError):
            generate_beta_params(5, 0)
        with pytest.raises(ValueError):
            generate_beta_params(5, 6)


class TestComputeCoxDirection(object):
    
    def test_sum_to_one(self):
        q = np.array([0.3,0.5,0.2])
        cox = compute_cox_direction(q,0)
        
        assert np.allclose(np.sum(cox,axis=1),np.ones(30))
        

class TestGetChoiceProbabilities(object):
    

    def test_2D_array(self):
        # Test that the function returns a 2-D numpy array
        U = np.array([[1, 2, 3], [4, 5, 6]],dtype=np.double)
        P = get_choice_probabilities(U)
        assert isinstance(P, np.ndarray)
        assert P.ndim == 2
    
    
    def test_shape(self):
        # Test that the shape of the returned array is correct
        U = np.array([[1, 2, 3], [4, 5, 6]],dtype=np.double)
        P = get_choice_probabilities(U)
        assert P.shape == (2, 3)
    
    
    def test_sum_to_one(self):
        # Test that the sum of the probabilities for each decision is 1
        U = np.array([[1, 2, 3], [4, 5, 6]],dtype=np.double)
        P = get_choice_probabilities(U)
        assert np.allclose(np.sum(P, axis=0), np.ones(3))
    
    def test_calculate(self):
        # Test that the probabilities are calculated correctly
        U = np.array([[1, 2, 3], [4, 5, 6]],dtype=np.double)
        P = get_choice_probabilities(U)
        assert np.allclose(P, np.array([[0.04742587, 0.04742587, 0.04742587],[0.95257413, 0.95257413, 0.95257413]]))
    
    
class TestIOptimality(object):
    
    def test_design_01(self):
        
        design = read_csv_file("Tests/data/design_01.csv")
        beta = np.array( (1.36, 1.57, 2.47, -0.43, 0.50, 1.09))
        I_opt_value = get_i_optimality_mnl(design,3,beta)
        expected_value = 1.193834
        assert np.isclose(np.exp(I_opt_value), expected_value, rtol=1e-2)
    
    def test_design_02(self):
        
        design = read_csv_file("Tests/data/design_02.csv")
        beta = np.array( (1.36, 1.57, 2.47, -0.43, 0.50, 1.09))
        I_opt_value = get_i_optimality_mnl(design,3,beta)
        expected_value = 1.033
        assert np.isclose(np.exp(I_opt_value), expected_value, rtol=1e-2)
    
    def test_design_03(self):
        
        design = read_csv_file("Tests/data/design_03.csv")
        
        beta = np.array( (1.36, 1.57, 2.47, -0.43, 0.50, 1.09))
        I_opt_value = get_i_optimality_mnl(design,3,beta)
        expected_value = 0.5029075
        assert np.isclose(np.exp(I_opt_value), expected_value, rtol=1e-2)
    
    def test_design_04(self):
        
        design = read_csv_file("Tests/data/design_04.csv")
        
        beta = np.array((0.86, 0.21, 3.07, 2.34, 3.24, -20.59))
        I_opt_value = get_i_optimality_mnl(design,3,beta)
        expected_value = 2.28
        assert np.isclose(np.exp(I_opt_value), expected_value, rtol=1e-2)
    
    def test_design_05(self):
        
        design = read_csv_file("Tests/data/design_05.csv")
        
        beta = np.array((0.86, 0.21, 3.07, 2.34, 3.24, -20.59))
        I_opt_value = get_i_optimality_mnl(design,3,beta)
        expected_value = 2.30
        assert np.isclose(np.exp(I_opt_value), expected_value, rtol=1e-2)   
        
    def test_design_06(self):
        
        design = read_csv_file("Tests/data/design_06.csv")
        
        beta = np.array((0.86, 0.21, 3.07, 2.34, 3.24, -20.59))
        I_opt_value = get_i_optimality_mnl(design,3,beta)
        expected_value = 1.035
        assert np.isclose(np.exp(I_opt_value), expected_value, rtol=1e-2)

class TestDOptimality(object):
    pass


class TestTransformVarcovMatrix(object):
    
    def test_case_1(self):
        # Test case 1: q = 1
        id_matrix = np.identity(5)
        q = 1
        expected_result = np.array([[1., 0., 0., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
        
        result = transform_varcov_matrix(id_matrix, q)
        assert np.array_equal(result, expected_result), f"Test case 1 failed: {result}"
    
    def test_case_2(self):
        # Test case 2: q = 2
        id_matrix = np.identity(6)
        q = 2
        expected_result = np.array([[2., 0., 0., 0., 0.], 
                                    [0., 1., 0., 0., 0.], 
                                    [0., 0., 1., 0., 0.], 
                                    [0., 0., 0., 1., 0.], 
                                    [0., 0., 0., 0., 1.]])
        result = transform_varcov_matrix(id_matrix, q)
        assert np.array_equal(result, expected_result), f"Test case 2 failed: {result}"

    def test_case_3(self):
        # Test case 3: q = 3
        id_matrix = np.identity(4)
        q = 3
        expected_result = np.array([[2., 1., 0.],
                                    [1., 2., 0.],
                                    [0., 0., 1.]])
        result = transform_varcov_matrix(id_matrix, q)
        assert np.array_equal(result, expected_result), f"Test case 3 failed: {result}"

    def test_case_4(self):
        # Test case 4: q = 4 (all rows and columns except the last)
        id_matrix = np.identity(5)
        q = 4
        expected_result = np.array([[2., 1., 1., 0.],
                                    [1., 2., 1., 0.],
                                    [1., 1., 2., 0.],
                                    [0., 0., 0., 1.]])
        result = transform_varcov_matrix(id_matrix, q)
        assert np.array_equal(result, expected_result), f"Test case 4 failed: {result}"

    


    def test_transform_varcov_matrix_case1(self):
        # Test case 1: identity matrix of size 4, q = 2, k = 1
        id_matrix = np.identity(4)
        q = 2
        k = 1
        expected_output = np.array([[2., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]])
        assert np.array_equal(transform_varcov_matrix(id_matrix, q, k), expected_output)


    def test_transform_varcov_matrix_case2(self):
        # Test case 2: identity matrix of size 3, q = 1, k = 2
        id_matrix = np.identity(3)
        q = 1
        k = 2
        expected_output = np.array([[2., 0.],       
                                    [0., 2.]])
        assert np.array_equal(transform_varcov_matrix(id_matrix, q, k), expected_output)


    def test_transform_varcov_matrix_case3(self):
        # Test case 3: identity matrix of size 2, q = 2, k = 0.5
        id_matrix = np.identity(2)
        q = 2
        k = 0.5
        expected_output = np.array([[1.]])
        assert np.array_equal(transform_varcov_matrix(id_matrix, q, k), expected_output)

