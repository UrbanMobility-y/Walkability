"""
Behavioral Modeling and Policy Scenario Simulation Module
Agent-based simulation for pedestrian routing and policy intervention analysis.
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, spearmanr
import random

class SegmentCostFunction:
    """
    Computes exponentially-penalized traversal costs.
    Ref: Eq. 14 in Supplementary Note.
    """
    def __init__(self, beta=1.8):
        self.beta = beta

    def compute_segment_cost(self, segment_length_m, walkability_index):
        # WI must be clamped to [0, 1]
        wi = np.clip(walkability_index, 0.0, 1.0)
        # Cost = L * exp(beta * (1 - WI))
        return segment_length_m * np.exp(self.beta * (1.0 - wi))


class RouteOptimization:
    """
    Finds minimum-cost routes and computes experiences based on Composite Index.
    """
    def __init__(self, street_network):
        self.network = street_network

    def find_minimum_cost_route(self, origin, dest, cost_function, sigma=0.55, time_period=None):
        """
        Calculates optimal path.
        Step 1: Calculate Segment WI = (1-sigma)SWI + sigma*DWI
        Step 2: Calculate Segment Cost = L * exp(beta*(1-WI))
        Step 3: Dijkstra
        """
        weighted_graph = nx.DiGraph()

        # Optimization: Iterate only necessary edges if graph is huge, 
        # but here assuming standard networkx usage
        for u, v, data in self.network.edges(data=True):
            length = data.get('length_m', 100.0)
            swi = data.get('SWI', 0.5)

            # Get dynamic DWI
            if time_period:
                dwi = data.get(f'DWI_{time_period}', data.get('DWI', 0.5))
            else:
                dwi = data.get('DWI', 0.5)

            # Calculate Segment Composite Walkability (Eq. 13)
            wi = (1.0 - sigma) * swi + sigma * dwi

            # Calculate Edge Weight (Eq. 14)
            weight = cost_function.compute_segment_cost(length, wi)
            weighted_graph.add_edge(u, v, weight=weight)

        try:
            path = nx.dijkstra_path(weighted_graph, origin, dest, weight='weight')
            segments = [(path[i], path[i+1]) for i in range(len(path) - 1)]
            return path, segments
        except nx.NetworkXNoPath:
            return None, None

    def compute_route_experience(self, route_segments, metric_type='static', 
                                 time_period=None, sigma=None):
        """
        Computes aggregated experience along the route.
        """
        values = []
        weights = [] # Time duration (proportional to length if speed constant)

        for u, v in route_segments:
            data = self.network.get_edge_data(u, v)
            if not data: continue

            l = data.get('length_m', 100.0)
            weights.append(l)

            # Extract basic metrics
            swi = data.get('SWI', 0.5)
            if time_period:
                dwi = data.get(f'DWI_{time_period}', data.get('DWI', 0.5))
            else:
                dwi = data.get('DWI', 0.5)

            # Logic Branching
            if metric_type == 'static':
                values.append(swi)
            elif metric_type == 'dynamic':
                values.append(dwi)
            elif metric_type == 'composite':
                # Critical: Calculate Segment Composite WI first
                if sigma is None:
                    raise ValueError("Sigma must be provided for composite metric")
                
                # WI_seg = (1-sigma)*SWI_seg + sigma*DWI_seg
                wi_seg = (1.0 - sigma) * swi + sigma * dwi
                values.append(wi_seg)

        if sum(weights) == 0:
            return 0.5

        return np.average(values, weights=weights)


class BehavioralModelCalibration:
    """Calibrates pedestrian routing model parameters (sigma, beta) to observed behavior"""

    def __init__(self, observed_trips, street_network):
        self.observed_trips = observed_trips
        self.network = street_network

    def grid_search_parameters(self, sigma_range=np.arange(0, 1.05, 0.05),
                              beta_range=np.arange(0, 5.1, 0.1),
                              sample_size=1000):
        """
        Performs exhaustive grid search over parameter space.
        Note: sample_size reduced to 1000 for performance.
        """

        # Sample calibration trips
        if len(self.observed_trips) > sample_size:
            calibration_trips = self.observed_trips.sample(n=sample_size, 
                                                         random_state=42)
        else:
            calibration_trips = self.observed_trips

        mse_values = np.zeros((len(sigma_range), len(beta_range)))

        for i, sigma in enumerate(sigma_range):
            for j, beta in enumerate(beta_range):
                # Compute MSE for this parameter pair
                mse = self._compute_mse_for_parameters(sigma, beta, 
                                                     calibration_trips)
                mse_values[i, j] = mse

        # Find optimal parameters
        min_idx = np.unravel_index(np.argmin(mse_values), mse_values.shape)
        optimal_params = {
            'sigma': sigma_range[min_idx[0]],
            'beta': beta_range[min_idx[1]],
            'min_mse': mse_values[min_idx]
        }

        return optimal_params, mse_values

    def _compute_mse_for_parameters(self, sigma, beta, calibration_trips):
        """Computes Mean Squared Error for a specific (sigma, beta) pair."""
        route_opt = RouteOptimization(self.network)
        cost_func = SegmentCostFunction(beta=beta)

        swi_errors = []
        dwi_errors = []

        for idx, trip in calibration_trips.iterrows():
            origin_node = trip.get('origin_node_id')
            dest_node = trip.get('destination_node_id')
            tp = trip.get('time_period') 

            # Simulate behavior with current parameters
            path, segments = route_opt.find_minimum_cost_route(
                origin_node, 
                dest_node, 
                cost_func, 
                sigma=sigma,        
                time_period=tp      
            )

            if path is None: continue

            # Compute metrics for the chosen path
            pred_swi = route_opt.compute_route_experience(segments, 'static')
            pred_dwi = route_opt.compute_route_experience(segments, 'dynamic', time_period=tp)

            # Compare to Ground Truth (Observed Experience)
            obs_swi = trip.get('observed_PE_SWI', 0.5)
            obs_dwi = trip.get('observed_PE_DWI', 0.5)

            swi_errors.append((pred_swi - obs_swi) ** 2)
            dwi_errors.append((pred_dwi - obs_dwi) ** 2)

        if len(swi_errors) == 0: return float('inf')

        # Combined MSE (Eq 16)
        return np.mean(swi_errors) + np.mean(dwi_errors)

    def validate_calibrated_model(self, optimal_sigma, optimal_beta, validation_trips=None):
        """Validates calibrated model on held-out test set."""
        if validation_trips is None:
            validation_trips = self.observed_trips

        route_opt = RouteOptimization(self.network)
        cost_func = SegmentCostFunction(beta=optimal_beta)

        predicted_swi, observed_swi = [], []
        predicted_dwi, observed_dwi = [], []

        for idx, trip in validation_trips.iterrows():
            o, d = trip.get('origin_node_id'), trip.get('destination_node_id')
            tp = trip.get('time_period')

            path, segments = route_opt.find_minimum_cost_route(
                o, d, cost_func, sigma=optimal_sigma, time_period=tp
            )

            if path is None: continue

            predicted_swi.append(route_opt.compute_route_experience(segments, 'static'))
            predicted_dwi.append(route_opt.compute_route_experience(segments, 'dynamic', time_period=tp))
            observed_swi.append(trip.get('observed_PE_SWI', 0.5))
            observed_dwi.append(trip.get('observed_PE_DWI', 0.5))

        return {
            'pearson_r_SWI': pearsonr(observed_swi, predicted_swi)[0],
            'spearman_rho_SWI': spearmanr(observed_swi, predicted_swi)[0],
            'pearson_r_DWI': pearsonr(observed_dwi, predicted_dwi)[0],
            'spearman_rho_DWI': spearmanr(observed_dwi, predicted_dwi)[0],
            'n_trips': len(observed_swi)
        }


class PolicyScenarioSimulator:
    """Simulates policy interventions via Agent-Based Re-routing."""
    
    def __init__(self, street_network, observed_trips, calibrated_params):
        self.base_graph = street_network
        self.trips = observed_trips
        self.sigma = calibrated_params['sigma']
        self.beta = calibrated_params['beta']
        self.cost_func = SegmentCostFunction(beta=self.beta)

    def _run_simulation_batch(self, graph, trip_set, override_dest=False):
        router = RouteOptimization(graph)
        results = []

        for idx, trip in trip_set.iterrows():
            o = trip['origin_node']
            d = trip['simulated_dest'] if override_dest else trip['dest_node']
            tp = trip.get('time_period')

            # 1. Re-route (Agent Behavior)
            path, segments = router.find_minimum_cost_route(
                o, d, self.cost_func, sigma=self.sigma, time_period=tp
            )

            if segments:
                # 2. Measure Composite Experience
                pe_comp = router.compute_route_experience(
                    segments, metric_type='composite', 
                    time_period=tp, sigma=self.sigma
                )
                results.append({
                    'PE_SWI': router.compute_route_experience(segments, 'static'),
                    'PE_DWI': router.compute_route_experience(segments, 'dynamic', time_period=tp),
                    'PE_Composite': pe_comp,
                    'pct_change_composite': 0.0 # Placeholder
                })

        if not results: return {}
        
        # Calculate mean stats
        df = pd.DataFrame(results)
        return df.mean().to_dict()

    def scenario_amenity_localization(self, alpha_values=np.arange(0, 1.1, 0.1)):
        """Scenario 1: Change Destination (Graph Unchanged)"""
        output = []
        for alpha in alpha_values:
            sim_trips = self.trips.copy()
            sim_trips['simulated_dest'] = sim_trips['dest_node']

            # Redirect fraction 'alpha' of long trips to local candidates
            mask = sim_trips['is_long_trip'] & (np.random.random(len(sim_trips)) < alpha)
            for idx in sim_trips[mask].index:
                cands = sim_trips.at[idx, 'local_candidates']
                if isinstance(cands, list) and cands:
                    sim_trips.at[idx, 'simulated_dest'] = random.choice(cands)

            res = self._run_simulation_batch(self.base_graph, sim_trips, override_dest=True)
            res['alpha'] = alpha
            output.append(res)
        return pd.DataFrame(output)

    def scenario_built_env_improvement(self, alpha_values=np.arange(0, 1.1, 0.1)):
        """Scenario 2: Change Graph SWI (Dest Unchanged)"""
        output = []
        for alpha in alpha_values:
            temp_graph = self.base_graph.copy()
            for u, v, data in temp_graph.edges(data=True):
                base = data.get('SWI', 0.5)
                # Improve SWI by alpha percent of the gap to 1.0
                data['SWI'] = base + (1.0 - base) * alpha

            res = self._run_simulation_batch(temp_graph, self.trips)
            res['alpha'] = alpha
            output.append(res)
        return pd.DataFrame(output)

    def scenario_traffic_calming(self, alpha_values=np.arange(0, 1.1, 0.1)):
        """Scenario 3: Change Graph DWI (Dest Unchanged)"""
        output = []
        for alpha in alpha_values:
            temp_graph = self.base_graph.copy()
            for u, v, data in temp_graph.edges(data=True):
                for key in data:
                    if key.startswith('DWI'):
                        base = data[key]
                        # Improve DWI (Higher is better)
                        data[key] = base + (1.0 - base) * alpha

            res = self._run_simulation_batch(temp_graph, self.trips)
            res['alpha'] = alpha
            output.append(res)
        return pd.DataFrame(output)