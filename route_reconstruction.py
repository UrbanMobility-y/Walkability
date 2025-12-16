"""
Route Reconstruction Module
Reconstructs detailed routes using AMap V5 APIs and performs high-precision metric segmentation.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import transform
import requests


class RouteReconstructor:
    """Interface for AMap Route Planning API (V5)."""

    def __init__(self, api_key):
        self.api_key = api_key
        # User specified to keep V5
        self.base_url = "https://restapi.amap.com/v5/direction"

    def _decode_polyline(self, polyline_str):
        """Decodes 'lon,lat;lon,lat' string to coordinate list."""
        if not polyline_str: return []
        coords = []
        for point_str in polyline_str.split(';'):
            lon, lat = map(float, point_str.split(','))
            coords.append((lon, lat))
        return coords

    def get_pedestrian_route(self, origin, dest):
        """
        Reconstructs walking route.
        Params: alternative_route=3, isindoor=0
        """
        params = {
            'key': self.api_key,
            'origin': f"{origin[0]},{origin[1]}",
            'destination': f"{dest[0]},{dest[1]}",
            'alternative_route': 3, 
            'isindoor': 0, 
            'show_fields': 'polyline,cost'
        }

        resp = requests.get(f"{self.base_url}/walking", params=params)
        data = resp.json()

        if data['info'] != 'OK' or not data['route']['paths']:
            return None

        # Select first (optimal) route
        path = data['route']['paths'][0]
        return {
            'geometry': LineString(self._decode_polyline(path['polyline'])),
            'distance_m': float(path['distance']),
            'duration_sec': float(path['cost']['duration']), # V5 structure
            'steps': path.get('steps', [])
        }

    def get_driving_route(self, origin, dest):
        """
        Reconstructs driving route.
        Params: strategy=2 (Regular fastest)
        """
        params = {
            'key': self.api_key,
            'origin': f"{origin[0]},{origin[1]}",
            'destination': f"{dest[0]},{dest[1]}",
            'strategy': 2, 
            'cartype': 0, 
            'show_fields': 'polyline,cost'
        }

        resp = requests.get(f"{self.base_url}/driving", params=params)
        data = resp.json()

        if data['info'] != 'OK' or not data['route']['paths']:
            return None

        path = data['route']['paths'][0]
        return {
            'geometry': LineString(self._decode_polyline(path['polyline'])),
            'distance_m': float(path['distance']),
            'duration_sec': float(path['cost']['duration']),
            'steps': path.get('steps', []),
            'toll_dist': float(path.get('cost', {}).get('toll_distance', 0)),
            'traffic_lights': len(path.get('steps', [])) 
        }


def get_projection_func(ref_lat):
    """
    Returns a transformer function to project (lon, lat) to (x_meters, y_meters).
    Uses Local Equirectangular Approximation centered at ref_lat.
    """
    # Constants for WGS84
    LAT_METERS_PER_DEGREE = 111320.0
    LON_METERS_PER_DEGREE = 111320.0 * np.cos(np.radians(ref_lat))

    def project(lon, lat):
        return (lon * LON_METERS_PER_DEGREE, lat * LAT_METERS_PER_DEGREE)

    return project

def project_geometry(geom, ref_lat):
    """Projects a Shapely geometry to the metric system."""
    project_func = get_projection_func(ref_lat)

    # shapely.ops.transform expects func(x, y, z=None)
    def transformer(x, y, z=None):
        nx, ny = project_func(x, y)
        return (nx, ny) if z is None else (nx, ny, z)

    return transform(transformer, geom)


def validate_route_geometry(gps_pings, route_geom, threshold_m=100, ratio=0.8):
    """
    Validates route using metric projection.
    """
    if len(gps_pings) < 2: return False, 0.0

    # 1. Establish Projection Center
    mean_lat = gps_pings[0][1]

    # 2. Project Route to Meters
    metric_route = project_geometry(route_geom, mean_lat)

    # 3. Project Pings and Check Distance
    match_count = 0
    proj_func = get_projection_func(mean_lat)

    for lon, lat in gps_pings:
        x, y = proj_func(lon, lat)
        pt = Point(x, y)

        # Calculate distance in pure meters
        dist = pt.distance(metric_route)

        if dist <= threshold_m:
            match_count += 1

    matched_ratio = match_count / len(gps_pings)
    return matched_ratio >= ratio, matched_ratio


def batch_reconstruct_trips(trips_df, gps_data_df, api_key, mode='pedestrian'):
    """Orchestrates reconstruction and validation."""
    reconstructor = RouteReconstructor(api_key)
    results = []

    for idx, trip in trips_df.iterrows():
        trip_id = trip['trip_id']
        origin = (trip['o_lon'], trip['o_lat'])
        dest = (trip['d_lon'], trip['d_lat'])

        # 1. Call API
        if mode == 'pedestrian':
            route = reconstructor.get_pedestrian_route(origin, dest)
        else:
            route = reconstructor.get_driving_route(origin, dest)

        if not route:
            continue

        # 2. Validation
        trip_pings = gps_data_df[gps_data_df['trip_id'] == trip_id][['lon', 'lat']].values
        is_valid, ratio = validate_route_geometry(trip_pings, route['geometry'])

        results.append({
            'trip_id': trip_id,
            'geometry': route['geometry'],
            'distance_m': route['distance_m'],
            'duration_sec': route['duration_sec'],
            'is_valid': is_valid,
            'match_ratio': ratio,
            'steps_data': route['steps']
        })

    return pd.DataFrame(results)


def segment_routes_optimized(routes_df, osm_network_gdf, buffer_size_m=20):
    """
    Maps routes to OSM segments.
    Projects intersection geometries to calculate precise metric lengths.
    """

    routes_gdf = gpd.GeoDataFrame(routes_df, geometry='geometry', crs="EPSG:4326")

    # 1. Create Search Buffer (Approximate degree buffer for efficient spatial join)
    mean_lat = routes_gdf.geometry.centroid.y.mean()
    # Safe buffer size estimation
    buffer_deg = buffer_size_m / (111000.0 * np.cos(np.radians(mean_lat)))

    routes_buffered = routes_gdf.copy()
    routes_buffered['geometry'] = routes_buffered.geometry.buffer(buffer_deg)

    # 2. Spatial Join (Filter Candidates)
    joined = gpd.sjoin(osm_network_gdf, routes_buffered, 
                       how="inner", predicate="intersects")

    final_segments = []

    # 3. Calculate Exact Metric Overlap
    # Group by Route to minimize projection overhead context switching
    for route_idx, group in joined.groupby('index_right'):
        route_geom = routes_gdf.loc[route_idx, 'geometry']

        # Determine local projection based on this specific route's latitude
        local_lat = route_geom.centroid.y

        # Project the whole route once
        metric_route = project_geometry(route_geom, local_lat)

        for osm_idx, row in group.iterrows():
            osm_geom = osm_network_gdf.loc[osm_idx, 'geometry']

            # Project OSM segment to the same metric system
            metric_osm = project_geometry(osm_geom, local_lat)

            # Calculate Intersection in Metric Space
            intersection = metric_route.intersection(metric_osm)

            if not intersection.is_empty and intersection.length > 1e-3:
                # Length is now directly in meters
                len_m = intersection.length

                final_segments.append({
                    'trip_id': routes_gdf.loc[route_idx, 'trip_id'],
                    'osm_segment_id': osm_idx,
                    'segment_type': row.get('highway', 'unknown'),
                    'overlap_length_m': len_m
                })

    return pd.DataFrame(final_segments)