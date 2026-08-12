import networkx as nx
import numpy as np
import pandas as pd

from neighborhood_walkability import (
    catchment_edge_keys,
    circuity_adjusted_radius_m,
    estimate_mean_circuity,
    length_weighted_mean,
    aggregate_fixed_grid_cells,
)


def test_city_radius_follows_circuity_equation():
    assert estimate_mean_circuity([120, 240], [100, 200]) == 1.2
    assert circuity_adjusted_radius_m(1.2) == 750


def test_geometric_catchment_does_not_select_every_edge():
    graph = nx.Graph()
    graph.add_node('A', x=0, y=0)
    graph.add_node('B', x=100, y=0)
    graph.add_node('C', x=300, y=0)
    graph.add_node('D', x=500, y=0)
    graph.add_node('E', x=600, y=0)
    graph.add_edge('A', 'B', length_m=100)
    graph.add_edge('B', 'C', length_m=200)
    graph.add_edge('D', 'E', length_m=100)
    selected = catchment_edge_keys(graph, {'home_node': 'A', 'catchment_radius_m': 125})
    assert ('A', 'B', None) in selected
    assert ('B', 'C', None) in selected  # partial intersection is included
    assert ('D', 'E', None) not in selected
    assert np.isclose(length_weighted_mean([0.2, 0.8], [1, 3]), 0.65)


def test_fixed_grid_aggregation_is_length_weighted_and_home_independent():
    pieces = pd.DataFrame({
        'cell_id': ['g1', 'g1'],
        'length_in_cell_m': [25, 75],
        'SWI': [0.2, 0.8],
        'DWI': [0.9, 0.5],
    })
    result = aggregate_fixed_grid_cells(pieces).iloc[0]
    assert np.isclose(result['Neighborhood_SWI'], 0.65)
    assert np.isclose(result['Neighborhood_DWI'], 0.60)
