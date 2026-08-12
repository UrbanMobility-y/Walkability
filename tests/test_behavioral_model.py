import networkx as nx
import pandas as pd
from behavioral_modeling_policy import (
    BehavioralModelCalibration,
    PolicyScenarioSimulator,
    RouteOptimizer,
    draw_stratified_calibration_sample,
)
from walkability_indices import PedestrianExperienceCalculator


def demo_graph():
    G = nx.DiGraph()
    G.add_node('A', x=0, y=0)
    G.add_node('B', x=100, y=0)
    G.add_node('C', x=200, y=0)
    G.add_edge('A','B', length_m=100, SWI=0.8, DWI=0.8)
    G.add_edge('B','C', length_m=100, SWI=0.8, DWI=0.8)
    G.add_edge('A','C', length_m=250, SWI=0.2, DWI=0.2)
    return G


def test_route_optimizer_avoids_low_walkability_when_beta_high():
    G = demo_graph()
    path = RouteOptimizer(G).route('A','C', sigma=0.5, beta=5)
    assert path == ['A','B','C']


def test_multidigraph_route_retains_selected_parallel_edge_key():
    graph = nx.MultiDiGraph()
    graph.add_edge('A', 'B', key=0, length_m=100, SWI=0.1, DWI=0.1)
    graph.add_edge('A', 'B', key=1, length_m=100, SWI=0.9, DWI=0.9)

    path = RouteOptimizer(graph).route('A', 'B', sigma=0.5, beta=5.0)
    assert path == ['A', 'B']
    assert path.edge_keys == (1,)
    assert path.keyed_edges == (('A', 'B', 1),)

    metrics = PedestrianExperienceCalculator(graph).compute_path_metrics(
        path, '2024-09-01 08:00'
    )
    assert metrics['PE_SWI'] == 0.9
    assert metrics['PE_DWI'] == 0.9


def test_policy_simulator_runs():
    G = demo_graph()
    trips = pd.DataFrame([{'origin_node':'A','dest_node':'C','departure_time':'2024-09-01 08:00','observed_PE_SWI':0.8,'observed_PE_DWI':0.8,'is_extra_neighborhood':False,'local_candidates':['B'],'home_node':'A','catchment_radius_m':125}])
    sim = PolicyScenarioSimulator(G, trips, sigma=0.5, beta=1.0)
    out = sim.scenario_traffic_calming(alpha_values=[0, 1])
    assert len(out) == 2
    assert 'delta_PE_composite_pct' in out.columns


def test_policy_requires_residential_catchment():
    trips = pd.DataFrame([{'origin_node':'A','dest_node':'C','departure_time':'2024-09-01 08:00','observed_PE_SWI':0.8,'observed_PE_DWI':0.8,'is_extra_neighborhood':False,'local_candidates':['B']}])
    sim = PolicyScenarioSimulator(demo_graph(), trips, sigma=0.5, beta=1.0)
    try:
        sim.scenario_built_environment_improvement(alpha_values=[1])
        assert False, 'missing catchment metadata should raise an error'
    except ValueError as exc:
        assert 'catchment' in str(exc)


def test_index_perturbation_changes_only_explicit_catchment_edges():
    graph = demo_graph()
    trips = pd.DataFrame([{
        'origin_node':'A', 'dest_node':'C', 'departure_time':'2024-09-01 08:00',
        'observed_PE_SWI':0.8, 'observed_PE_DWI':0.8,
        'is_extra_neighborhood':False, 'local_candidates':['B'],
        'catchment_edges':[('A', 'B')],
    }])
    sim = PolicyScenarioSimulator(graph, trips, sigma=0.5, beta=1.0)
    treated = sim._treated_graph(trips.iloc[0], alpha=1.0, dimension='SWI')
    assert treated['A']['B']['SWI'] == 1.0
    assert treated['B']['C']['SWI'] == graph['B']['C']['SWI']


def test_calibration_sampler_targets_distance_and_peak_strata():
    trips = pd.DataFrame({
        'route_length_m': ([250] * 250 + [750] * 250 + [1500] * 250 + [2500] * 250),
        'departure_time': pd.date_range('2024-09-01', periods=1000, freq='5min'),
    })
    sample = draw_stratified_calibration_sample(trips, sample_size=200, random_state=7)
    assert len(sample) == 200
    assert sample['route_length_m'].min() == 250
    assert sample['route_length_m'].max() == 2500
