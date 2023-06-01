import numpy as np
import plotly.graph_objects as go
import pytest
from Tests.utils import check_mnl_design_sum

from MixtureOptDesign.mnl_visual.ternary_plot import plot_ternary_design

# Define fixture for design data
@pytest.fixture
def design_data():
    data = [0.3897873, 0.6292498, 0.7009263, 0.4262842, 0.246073770, 0.48413014, 0.2913878, 0.7201046335,
            0.4982357, 0.007519965, 0.2505569, 0.1549670, 0.5415304, 0.3639989, 0.04235179, 0.3279013,
            0.2375383, 0.1180457, 0.2000333, 0.2569634, 0.749005428, 0.03588967, 0.3591402, 0.0009022414,
            0.2549352, 0.636080256, 0.4460236, 0.6877322, 0.2635403, 0.3727396, 0.44510098, 0.2701244,
            0.3726744, 0.2527044, 0.0990404, 0.3167524, 0.004920801, 0.47998018, 0.3494720, 0.2789931250,
            0.2468291, 0.356399779, 0.3034196, 0.1573008, 0.1949293, 0.2632615, 0.51254723, 0.4019743]
    return np.array(data).reshape((3, 2, 8))


# Test if the function creates a plotly graph object
def test_plot_ternary_design_returns_figure(design_data):
    fig = plot_ternary_design(design_data)
    assert isinstance(fig, go.Figure)


# Test if the function creates a scatterternary trace with the expected data
def test_plot_ternary_design_trace_data(design_data):
    fig = plot_ternary_design(design_data)
    trace = fig.data[0]
    assert isinstance(trace, go.Scatterternary)
    assert np.array_equal(trace.a, design_data[1].flatten())
    assert np.array_equal(trace.b, design_data[0].flatten())
    assert np.array_equal(trace.c, design_data[2].flatten())

def test_plot_ternary_design_valid_input():
    # Valid input
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

    # Test valid input
    plot_ternary_design(design)

def test_plot_ternary_design_invalid_input():
    # Test invalid input with ValueError
    invalid_design = np.array([
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        [
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0],
        ],
        [
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
            [0.9, 1.0, 1.5],  # Invalid value
        ],
    ])

    with pytest.raises(ValueError):
        #Invalid input: sum of each point's coordinates must be equal to 1
        plot_ternary_design(invalid_design)

