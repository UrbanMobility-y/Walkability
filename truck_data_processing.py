"""
Truck GPS Data Processing Module
Processes freight vehicle GPS trajectories from regulatory platform.

"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from shapely.geometry import Point
import geopandas as gpd


def compute_distance(lat1, lon1, lat2, lon2):
    """
    Computes Great Circle (Haversine) distance in meters.
    Essential for accurate long-distance truck trajectory analysis.
    """
    R = 6371000.0  # Earth radius in meters

    phi1, lambda1 = np.radians(lat1), np.radians(lon1)
    phi2, lambda2 = np.radians(lat2), np.radians(lon2)

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    a = np.sin(dphi / 2.0)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def preprocess_truck_gps(gps_records, max_consecutive_gap=60, max_speed_change=50,
                         spatial_outlier_threshold=500):
    """
    Cleans raw GPS stream: removes duplicates, outliers, and data gaps.
    """
    initial_cnt = len(gps_records)

    # 1. Format & Sort
    gps_records['timestamp'] = pd.to_datetime(gps_records['timestamp'])
    gps_records = gps_records.sort_values(['truck_id', 'timestamp'])

    # 2. Data Loss Filter (Gap > 60s)
    # Calculate time gaps per truck
    gps_records['time_gap'] = gps_records.groupby('truck_id')['timestamp'].diff().dt.total_seconds()

    # Identify trucks with >5% data loss
    def has_high_loss(group):
        loss_rate = (group['time_gap'] > max_consecutive_gap).mean()
        return loss_rate > 0.05

    truck_stats = gps_records.groupby('truck_id').apply(has_high_loss)
    valid_trucks = truck_stats[~truck_stats].index
    gps_records = gps_records[gps_records['truck_id'].isin(valid_trucks)].copy()

    # 3. Deduplication
    gps_records = gps_records.drop_duplicates(subset=['truck_id', 'timestamp'])

    # 4. Spatial Outlier Detection (Ping > 500m from both prev and next)
    # Vectorized shift
    g = gps_records.groupby('truck_id')
    lat = gps_records['latitude']
    lon = gps_records['longitude']

    prev_lat, prev_lon = g['latitude'].shift(1), g['longitude'].shift(1)
    next_lat, next_lon = g['latitude'].shift(-1), g['longitude'].shift(-1)

    dist_prev = compute_distance(lat, lon, prev_lat, prev_lon)
    dist_next = compute_distance(lat, lon, next_lat, next_lon)

    # Valid if close to EITHER prev OR next (outlier if far from BOTH)
    is_outlier = (dist_prev > spatial_outlier_threshold) & (dist_next > spatial_outlier_threshold)
    gps_records = gps_records[~is_outlier.fillna(False)]

    # 5. Physics Check (Speed Jump > 50km/h)
    speed_diff = gps_records.groupby('truck_id')['speed_kmh'].diff().abs()
    gps_records = gps_records[speed_diff <= max_speed_change]

    return gps_records.reset_index(drop=True)


def identify_truck_stops_adaptive(truck_trajectory):
    """
    Identifies stops using a Gaussian Mixture Model to find the speed threshold.
    """
    speeds = truck_trajectory['speed_kmh'].values
    speeds_clean = speeds[(speeds >= 0) & (speeds < 150)] # Sanity filter

    # Adaptive Thresholding using KDE saddle point
    try:
        kde = gaussian_kde(speeds_clean, bw_method=0.1)
        x_grid = np.linspace(0, 10, 100)
        density = kde(x_grid)

        # Find local minimum (saddle) between stationary peak (~0) and moving peak
        # Simplified: look for min in range [0.5, 5.0] km/h
        search_mask = (x_grid >= 0.5) & (x_grid <= 5.0)
        if search_mask.sum() > 0:
            local_min_idx = np.argmin(density[search_mask])
            threshold = x_grid[search_mask][local_min_idx]
        else:
            threshold = 2.0 # Fallback
    except:
        threshold = 3.0 # Fallback if KDE fails

    # Segment stationary periods
    is_stop = truck_trajectory['speed_kmh'] < threshold
    truck_trajectory['group'] = (is_stop != is_stop.shift()).cumsum()

    stops = []
    # Filter only groups that are stops
    stop_groups = truck_trajectory[is_stop].groupby('group')

    for _, group in stop_groups:
        start_t = group['timestamp'].min()
        end_t = group['timestamp'].max()
        duration = (end_t - start_t).total_seconds() / 60.0

        stops.append({
            'centroid_lat': group['latitude'].mean(),
            'centroid_lon': group['longitude'].mean(),
            'start_time': start_t,
            'end_time': end_t,
            'duration_minutes': duration
        })

    return pd.DataFrame(stops)


def identify_truck_trip_ods_recursive(all_stops, gps_trajectory, 
                                      long_duration_threshold=240, 
                                      short_duration_threshold=20,
                                      max_circuity=2.0,
                                      poi_database=None):
    """
    Identifies trips recursively by checking circuity between long stops.
    Includes Freight POI validation.
    """
    if all_stops.empty:
        return pd.DataFrame()

    # Sort data
    all_stops = all_stops.sort_values('start_time').reset_index(drop=True)
    gps_trajectory = gps_trajectory.sort_values('timestamp')

    # --- Context Setup (Closure Optimization) ---
    gps_timestamps = gps_trajectory['timestamp'].values

    # Pre-calculate path distances
    lat_rad = np.radians(gps_trajectory['latitude'])
    dlat = gps_trajectory['latitude'].diff() * 111000
    dlon = gps_trajectory['longitude'].diff() * 111000 * np.cos(lat_rad)
    step_dist = np.sqrt(dlat**2 + dlon**2).fillna(0)
    gps_cum_dists = step_dist.cumsum().values

    # Pre-build Spatial Index for Validation
    poi_sindex = poi_database.sindex if (poi_database is not None) else None

    # --- Inner Helper: Validation ---
    def _is_valid_freight_location(stop):
        if poi_sindex is None: return True
        p = Point(stop['centroid_lon'], stop['centroid_lat'])
        # Buffer ~200m
        hits = list(poi_sindex.intersection(p.buffer(0.002).bounds))
        if not hits: return False

        matches = poi_database.iloc[hits]
        freight_cats = ['warehouse', 'factory', 'logistics', 'industrial']
        return matches['category'].isin(freight_cats).any()

    # --- Inner Helper: Recursive Split ---
    def _recurse(origin, dest, candidates):
        # 1. Calc Path Distance
        t0, t1 = origin['end_time'], dest['start_time']
        idx0 = np.searchsorted(gps_timestamps, t0)
        idx1 = np.searchsorted(gps_timestamps, t1)

        path_len = 0.0
        if idx1 > idx0:
            safe_idx1 = min(idx1, len(gps_cum_dists)-1)
            safe_idx0 = min(idx0, len(gps_cum_dists)-1)
            path_len = gps_cum_dists[safe_idx1] - gps_cum_dists[safe_idx0]

        # 2. Calc Euclidean Distance
        euc_len = compute_distance(origin['centroid_lat'], origin['centroid_lon'],
                                   dest['centroid_lat'], dest['centroid_lon'])

        circuity = path_len / euc_len if euc_len > 100 else 1.0

        # 3. Decision
        # Base Case: Direct trip or no intermediate stops to split
        if circuity <= max_circuity or candidates.empty:
            if _is_valid_freight_location(origin) and _is_valid_freight_location(dest):
                return [{
                    'origin_lat': origin['centroid_lat'],
                    'origin_lon': origin['centroid_lon'],
                    'dest_lat': dest['centroid_lat'],
                    'dest_lon': dest['centroid_lon'],
                    'start_time': t0,
                    'end_time': t1,
                    'distance_m': path_len,
                    'circuity': circuity
                }]
            return []

        # Recursive Case: Split at longest intermediate stop
        pivot_idx = candidates['duration_minutes'].idxmax()
        pivot = candidates.loc[pivot_idx]

        left_cands = candidates[candidates['end_time'] <= pivot['start_time']]
        right_cands = candidates[candidates['start_time'] >= pivot['end_time']]

        return _recurse(origin, pivot, left_cands) + _recurse(pivot, dest, right_cands)

    # --- Main Logic ---
    primary_stops = all_stops[all_stops['duration_minutes'] >= long_duration_threshold]
    if len(primary_stops) < 2: return pd.DataFrame()

    primary_stops = primary_stops.sort_values('start_time')
    trips = []

    for i in range(len(primary_stops) - 1):
        o = primary_stops.iloc[i]
        d = primary_stops.iloc[i+1]

        # Candidates are short stops strictly between O and D
        mask = (all_stops['start_time'] >= o['end_time']) & \
               (all_stops['end_time'] <= d['start_time']) & \
               (all_stops['duration_minutes'] >= short_duration_threshold) & \
               (all_stops['duration_minutes'] < long_duration_threshold)

        sub_trips = _recurse(o, d, all_stops[mask])
        trips.extend(sub_trips)

    return pd.DataFrame(trips)


def filter_on_road_stops(trip_ods, road_network):
    """
    Removes trip ends that fall directly on major roads (false positives due to congestion).
    """
    # Create GeoDataFrame for Origins and Destinations
    origins = gpd.GeoDataFrame(
        trip_ods, geometry=gpd.points_from_xy(trip_ods.origin_lon, trip_ods.origin_lat)
    )
    dests = gpd.GeoDataFrame(
        trip_ods, geometry=gpd.points_from_xy(trip_ods.dest_lon, trip_ods.dest_lat)
    )

    major_roads = road_network[
        road_network['highway'].isin(['motorway', 'trunk', 'primary'])
    ]

    # Buffer roads (e.g., 20m)
    # Note: sindex queries are used for efficiency
    road_buffer = major_roads.geometry.buffer(20 / 111000.0).unary_union

    # Filter
    valid_o = ~origins.geometry.intersects(road_buffer)
    valid_d = ~dests.geometry.intersects(road_buffer)

    return trip_ods[valid_o & valid_d].copy()