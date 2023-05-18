import plotly.graph_objects as go
import numpy as np

from Tests.utils import check_mnl_design_sum

def plot_ternary_design(design):
    """
    Plot a ternary design in a 3D scatter plot.

    Parameters
    ----------
    design : ndarray
        A numpy array of shape (3, 2, N) containing the N design points.
        The first dimension corresponds to the three ingredients,
        the second dimension corresponds to the two vertices of the ternary plot,
        and the third dimension corresponds to the N design points.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object.

    Examples
    --------
    >>> import numpy as np
    >>> from mypackage.visualisation import plot_ternary_design
    >>> design = np.random.rand(3, 2, 10)
    >>> fig = plot_ternary_design(design)
    >>> fig.show()

    Notes
    -----
    This function requires the `plotly` and `numpy` packages to be installed.

    The ternary plot is a triangular graph where the three vertices represent
    the three ingredients of a mixture, and the points inside the triangle represent
    the different compositions of the mixture.

    The function uses Plotly to create an interactive 3D scatter plot of the design points,
    where each point is represented by a circle marker with a blue color.
    The size of the marker is set to 7, and the points are not clipped at the boundary
    of the ternary plot, so they are fully visible even if they cross the boundary.
    The axis titles of the ternary plot are set to 'Ingredient 1', 'Ingredient 2',
    and 'Ingredient 3'.
    
    """
    #Check if Sum of ingredients in mixture is close to 1
    check_mnl_design_sum(design)
    
    # Flatten design points
    ingredient1 = abs(design[0].flatten())
    ingredient2 = abs(design[1].flatten())
    ingredient3 = abs(design[2].flatten())

    # Create figure object
    fig = go.Figure()

    # Define scatter trace for each design point
    fig.add_trace(go.Scatterternary(
        a=ingredient2,
        b=ingredient1,
        c=ingredient3,
        mode='markers',
        marker=dict(
            symbol='circle',
            size=7,
            color='blue',
        ),
         name='Design Points',
        text=[f'Ing1: {q1:.2f}, Ing2: {q2:.2f}, Ing3: {q3:.2f}' for q1, q2, q3 in zip(ingredient1, ingredient2, ingredient3)],  # Add custom hover text
        hoverinfo='text',  # Show only custom hover text
        cliponaxis=False,  # Make points fully visible
    ))
      
    # Define layout
    fig.update_layout(
        ternary=dict(
            sum=1,
            aaxis_title='Ingredient 2',
            baxis_title='Ingredient 1',
            caxis_title='Ingredient 3'
        ),
        showlegend=True,
        width=600,
        height=600
    )
   # Show the plot
    return fig
