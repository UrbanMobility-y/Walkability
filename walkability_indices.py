"""
Walkability Index Calculation Module
Computes static and dynamic walkability indices at street, neighborhood, and trip scales.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import interp1d
from datetime import timedelta


class StaticWalkabilityIndex:
    """Computes street-level static walkability index (SWI) based on 8 fixed indicators."""

    def __init__(self):
        # Weights defined in Supplementary Table 4
        self.weights = {
            'road_hierarchy': 0.3,
            'gradient': 0.3,
            'building_density': 0.1,
            'amenity_availability': 0.3,
            'transit_accessibility': 0.3,
            'greenness_ratio': 0.3,
            'industrial_ratio': 0.3,
            'water_proximity': 0.4
        }
        self.weight_sum = sum(self.weights.values()) # Should be 2.3

    def _score_hierarchy(self, road_type):
        mapping = {
            'motorway': 0.0, 'trunk': 0.0, 'primary': 0.0,
            'secondary': 0.2, 'tertiary': 0.5,
            'residential': 0.8, 'service': 0.85, 
            'living_street': 0.9, 'footway': 1.0, 'pedestrian': 1.0
        }
        return mapping.get(road_type, 0.5)

    def _score_gradient(self, slope_pct):
        slope = abs(slope_pct)
        if slope <= 3: return 1.0
        elif slope <= 6: return 0.7
        elif slope <= 12: return 0.5
        return 0.25

    def _score_building_density(self, ratio):
        # Inverse score as per methodology (Lower density preferred in this specific index)
        if ratio == 0: return 1.0
        elif ratio <= 0.2: return 0.8
        elif ratio <= 0.4: return 0.6
        elif ratio <= 0.6: return 0.4
        elif ratio <= 0.8: return 0.2
        return 0.0

    def _score_greenness(self, ratio):
        if ratio == 0: return 0.0
        elif ratio <= 0.05: return 0.7
        elif ratio <= 0.50: return 0.8
        elif ratio <= 0.75: return 0.9
        return 1.0

    def _score_industrial(self, ratio):
        if ratio == 0: return 1.0
        elif ratio <= 0.05: return 0.3
        elif ratio <= 0.50: return 0.2
        elif ratio <= 0.75: return 0.1
        return 0.0

    def compute_swi(self, segment_attr):
        """
        Computes SWI for a single segment.
        segment_attr: dict containing raw environmental data.
        """
        scores = {
            'road_hierarchy': self._score_hierarchy(segment_attr.get('highway', 'residential')),
            'gradient': self._score_gradient(segment_attr.get('gradient', 0)),
            'building_density': self._score_building_density(segment_attr.get('bldg_density', 0)),
            'amenity_availability': 1.0 if segment_attr.get('has_amenity') else 0.0,
            'transit_accessibility': 1.0 if segment_attr.get('has_transit') else 0.0,
            'greenness_ratio': self._score_greenness(segment_attr.get('green_ratio', 0)),
            'industrial_ratio': self._score_industrial(segment_attr.get('ind_ratio', 0)),
            'water_proximity': 1.0 if segment_attr.get('near_water') else 0.0
        }

        weighted_sum = sum(scores[k] * self.weights[k] for k in scores)
        return np.clip(weighted_sum / self.weight_sum, 0, 1)


class DynamicWalkabilityIndex:
    """Handles Traffic Stress, Quantile Normalization, and Crossing Indices."""

    def __init__(self):
        # Ref: CJJ 37-2012 Standard
        self.pcu_weights = {'car': 1.0, 'bus': 2.0, 'truck': 2.5}
        self.ecdf_func = None

    def compute_traffic_stress(self, car_density, bus_density, truck_density):
        """Calculates Traffic Stress (TS) in PCE/km."""
        return (car_density * self.pcu_weights['car'] +
                bus_density * self.pcu_weights['bus'] +
                truck_density * self.pcu_weights['truck'])

    def fit_quantile_normalizer(self, all_stress_values):
        """
        Fits the ECDF on the city-wide distribution of traffic stress.
        Ref: Note 3.2.4 Index Formulation.
        """
        sorted_ts = np.sort(all_stress_values)
        n = len(sorted_ts)
        # Empirical Cumulative Distribution Function
        y = np.arange(1, n + 1) / n

        # Create interpolation function for mapping TS -> Quantile
        self.ecdf_func = interp1d(sorted_ts, y, bounds_error=False, 
                                  fill_value=(0, 1), kind='linear')

    def get_dwi(self, traffic_stress):
        """Returns DWI = 1 - Quantile(TS). Higher is better."""
        if self.ecdf_func is None:
            raise ValueError("Normalizer not fitted. Call fit_quantile_normalizer first.")
        quantile = self.ecdf_func(traffic_stress)
        return 1.0 - quantile

    def compute_dci(self, connected_dwis):
        """
        Dynamic Crossing Index (DCI) for intersections.
        Ref: Note 3.3 Barrier Effect. DCI = Min(DWI of connected segments).
        """
        if not connected_dwis:
            return 0.5 # Default fallback
        return np.min(connected_dwis)


class NeighborhoodAggregator:
    """Calculates Home_SWI and Home_DWI based on 15-min catchment."""

    def compute_indices(self, home_geom, city_radius_m, segments_gdf, time_column=None):
        """
        Computes length-weighted averages for segments within the radius.
        Uses cos(lat) correction for circular buffer in geographic coordinates.
        """
        lat_correction = np.cos(np.radians(home_geom.y))

        # Buffer distance in degrees
        # Lat: 1 deg ~= 111000m
        # Lon: 1 deg ~= 111000m * cos(lat)
        buffer_dist_lat = city_radius_m / 111000.0
        buffer_dist_lon = city_radius_m / (111000.0 * lat_correction)

        # Create an approximate bounding box or simply use the larger dimension 
        # to ensure coverage, then filter by exact distance if needed.
        # Here we use the max to ensure the circle is fully contained within selection
        search_radius_deg = max(buffer_dist_lat, buffer_dist_lon)

        # 1. Spatial Filter
        buffer = home_geom.buffer(search_radius_deg) 
        nearby = segments_gdf[segments_gdf.intersects(buffer)]

        if nearby.empty:
            return np.nan

        # 2. Length-Weighted Average
        total_len = nearby['length_m'].sum()
        if total_len == 0: return np.nan

        home_swi = (nearby['SWI'] * nearby['length_m']).sum() / total_len

        home_dwi = np.nan
        if time_column and time_column in nearby.columns:
            home_dwi = (nearby[time_column] * nearby['length_m']).sum() / total_len

        return home_swi, home_dwi


class PedestrianExperienceCalculator:
    """Computes PE metrics with precise time-progression along the route."""

    def __init__(self, street_network):
        """
        street_network: networkx.Graph
        Nodes must have: 'type' (arterial/local), 'dci_schedule' (dict).
        Edges must have: 'length_m', 'SWI', 'dwi_schedule' (dict).
        """
        self.network = street_network
        self.walk_speed = 1.0 # m/s (Ref Note 4.1)

    def _get_time_key(self, dt):
        """Rounds datetime to nearest 5-min bucket string 'HH:MM'."""
        m = (dt.minute // 5) * 5
        return f"{dt.hour:02d}:{m:02d}"

    def _get_intersection_delay(self, node_id):
        """Estimates crossing delay based on road hierarchy."""
        node_data = self.network.nodes[node_id]
        level = node_data.get('road_level', 3)
        if level == 1: return 61.0
        elif level == 2: return 37.5
        else: return 8.0

    def compute_trip_metrics(self, path_nodes, start_time):
        """
        Calculates PE_SWI and PE_DWI by simulating the walk.
        """
        current_time = pd.to_datetime(start_time)

        accumulated_swi = 0.0
        accumulated_dwi = 0.0
        total_duration = 0.0      # Includes walk + wait
        total_walk_time = 0.0     # Only walking time (for SWI denominator)

        # Iterate through path (Node -> Edge -> Node)
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]

            # --- 1. Edge Traversal (Walking) ---
            edge_data = self.network.get_edge_data(u, v)
            length = edge_data.get('length_m', 100.0)
            walk_time = length / self.walk_speed

            # SWI (Static)
            swi = edge_data.get('SWI', 0.5)
            accumulated_swi += swi * walk_time

            # DWI (Dynamic) - Fetch based on arrival time at segment
            time_key = self._get_time_key(current_time)
            dwi_schedule = edge_data.get('dwi_schedule', {})
            dwi = dwi_schedule.get(time_key, edge_data.get('DWI', 0.5))
            accumulated_dwi += dwi * walk_time

            # Advance Time
            total_duration += walk_time
            total_walk_time += walk_time
            current_time += timedelta(seconds=walk_time)

            # --- 2. Intersection Crossing (Waiting) ---
            # Determine if 'v' is intermediate (requires crossing)
            if i < len(path_nodes) - 2:
                dwell_time = self._get_intersection_delay(v)

                # DCI (Dynamic) - Fetch based on arrival time at intersection
                time_key_int = self._get_time_key(current_time)
                node_data = self.network.nodes[v]
                dci_schedule = node_data.get('dci_schedule', {})
                dci = dci_schedule.get(time_key_int, 0.5)

                # Accumulate Intersection Experience into DWI
                accumulated_dwi += dci * dwell_time

                # Advance Time
                total_duration += dwell_time
                current_time += timedelta(seconds=dwell_time)

        if total_duration == 0:
            return {'PE_SWI': 0.5, 'PE_DWI': 0.5, 'duration': 0}

        pe_swi = accumulated_swi / total_walk_time if total_walk_time > 0 else 0.5
        pe_dwi = accumulated_dwi / total_duration

        return {
            'PE_SWI': pe_swi,
            'PE_DWI': pe_dwi,
            'duration_sec': total_duration
        }