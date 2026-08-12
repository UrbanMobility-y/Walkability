import networkx as nx
import numpy as np
import pytest
from walkability_indices import StaticWalkabilityIndex, DynamicWalkabilityIndex, PedestrianExperienceCalculator


def test_swi_range_and_equal_weights():
    seg = {'highway':'residential','gradient':2,'bldg_density':0.3,'has_amenity':True,'has_transit':False,'green_ratio':0.1,'industrial_ratio':0,'near_water':False}
    assert 0 <= StaticWalkabilityIndex().compute_swi(seg) <= 1
    assert 0 <= StaticWalkabilityIndex(StaticWalkabilityIndex.equal_weights()).compute_swi(seg) <= 1


def test_final_swi_is_city_ecdf_percentile():
    calc = StaticWalkabilityIndex().fit_quantile_normalizer([0.2, 0.4, 0.8, 0.9])
    np.testing.assert_allclose(calc.get_swi([0.4, 0.9]), [0.5, 1.0])


def test_dwi_quantile_and_dci():
    calc = DynamicWalkabilityIndex()
    ve = calc.compute_vehicular_exposure([1,2,3], [0,1,0], [0,0,1])
    calc.fit_quantile_normalizer(ve)
    dwi = calc.get_dwi(ve)
    assert len(dwi) == 3
    assert calc.compute_dci([0.8, 0.3, 0.7]) == 0.3


def test_static_experience_excludes_intersection_dwell_from_denominator():
    graph = nx.DiGraph()
    graph.add_node('A')
    graph.add_node('B', dwell_time_s=100, DCI=0.2)
    graph.add_node('C')
    graph.add_edge('A', 'B', length_m=100, SWI=0.8, DWI=0.8)
    graph.add_edge('B', 'C', length_m=100, SWI=0.8, DWI=0.8)
    metrics = PedestrianExperienceCalculator(graph).compute_path_metrics(
        ['A', 'B', 'C'], '2024-09-01 08:00'
    )
    assert metrics['PE_SWI'] == 0.8
    assert metrics['PE_DWI'] < metrics['PE_SWI']


def test_parallel_edges_require_keys_for_an_external_node_path():
    graph = nx.MultiDiGraph()
    graph.add_edge('A', 'B', key=0, length_m=100, SWI=0.1, DWI=0.1)
    graph.add_edge('A', 'B', key=1, length_m=100, SWI=0.9, DWI=0.9)
    calculator = PedestrianExperienceCalculator(graph)

    with pytest.raises(ValueError, match='parallel edges'):
        calculator.compute_path_metrics(['A', 'B'], '2024-09-01 08:00')

    metrics = calculator.compute_path_metrics(
        ['A', 'B'], '2024-09-01 08:00', edge_keys=[1]
    )
    assert metrics['PE_SWI'] == 0.9
    assert metrics['PE_DWI'] == 0.9
