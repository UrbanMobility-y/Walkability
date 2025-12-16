"""
Mobile Phone Data Processing Module
Handles anonymized mobile phone dataset preprocessing, stay detection, and trip OD identification
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point, LineString
from datetime import datetime, timedelta


def preprocess_raw_gps_data(gps_records, min_monthly_records=300):
    """
    Preprocesses raw GPS pings to enhance dataset integrity
    
    Parameters:
    -----------
    gps_records : DataFrame
        Columns: [user_id, timestamp, longitude, latitude, city_code]
    min_monthly_records : int
        Minimum number of records per month to retain user
        
    Returns:
    --------
    processed_records : DataFrame
        Cleaned GPS records with quality control applied
    statistics : dict
        Summary statistics of preprocessing
    """
    
    initial_users = gps_records['user_id'].nunique()
    initial_records = len(gps_records)
    
    # Remove duplicate records
    gps_records = gps_records.drop_duplicates(
        subset=['user_id', 'timestamp', 'longitude', 'latitude']
    )
    duplicates_removed = initial_records - len(gps_records)
    
    # Exclude single-city residents filter (users with records in >1 city)
    user_cities = gps_records.groupby('user_id')['city_code'].nunique()
    single_city_users = user_cities[user_cities == 1].index
    gps_records = gps_records[gps_records['user_id'].isin(single_city_users)]
    
    # Count monthly records per user
    gps_records['year_month'] = pd.to_datetime(gps_records['timestamp']).dt.to_period('M')
    monthly_counts = gps_records.groupby('user_id')['year_month'].count()
    qualified_users = monthly_counts[monthly_counts >= min_monthly_records].index
    gps_records = gps_records[gps_records['user_id'].isin(qualified_users)]
    
    final_users = gps_records['user_id'].nunique()
    final_records = len(gps_records)
    
    statistics = {
        'initial_users': initial_users,
        'initial_records': initial_records,
        'duplicates_removed': duplicates_removed,
        'users_after_preprocessing': final_users,
        'records_after_preprocessing': final_records,
        'retention_rate_users': final_users / initial_users,
        'retention_rate_records': final_records / initial_records
    }
    
    return gps_records, statistics


def identify_stays_dbscan(user_trajectory, eps=50, min_pts=10):
    """
    Identifies stationary periods (stays) in user trajectory using DBSCAN
    
    Parameters:
    -----------
    user_trajectory : DataFrame
        User's GPS pings sorted by timestamp
        Columns: [timestamp, longitude, latitude]
    eps : float
        Spatial search radius in meters
    min_pts : int
        Minimum number of points to form a cluster
        
    Returns:
    --------
    stays : DataFrame
        Identified stay locations with cluster centroids
        Columns: [centroid_lon, centroid_lat, num_pings, duration_minutes]
    """
    
    # Convert coordinates to meters for distance calculation
    coords_rad = np.radians(user_trajectory[['latitude', 'longitude']].values)
    
    # Haversine distance matrix (simplified for small areas)
    lat_lon = user_trajectory[['latitude', 'longitude']].values
    X = lat_lon.copy()
    
    # Run DBSCAN
    clustering = DBSCAN(eps=eps/111000, min_samples=min_pts).fit(X)
    user_trajectory['cluster'] = clustering.labels_
    
    # Extract stays (cluster_id != -1 represents actual clusters)
    stays_list = []
    for cluster_id in user_trajectory['cluster'].unique():
        if cluster_id != -1:  # Ignore noise points
            cluster_points = user_trajectory[user_trajectory['cluster'] == cluster_id]
            
            centroid_lat = cluster_points['latitude'].mean()
            centroid_lon = cluster_points['longitude'].mean()
            num_pings = len(cluster_points)
            
            time_start = pd.to_datetime(cluster_points['timestamp'].iloc[0])
            time_end = pd.to_datetime(cluster_points['timestamp'].iloc[-1])
            duration_minutes = (time_end - time_start).total_seconds() / 60
            
            stays_list.append({
                'centroid_lon': centroid_lon,
                'centroid_lat': centroid_lat,
                'num_pings': num_pings,
                'duration_minutes': duration_minutes,
                'start_time': time_start,
                'end_time': time_end
            })
    
    stays = pd.DataFrame(stays_list)
    return stays


def map_stays_to_poi(stays, poi_database, poi_buffer=100):
    """
    Maps identified stay clusters to nearest Points of Interest
    
    Parameters:
    -----------
    stays : DataFrame
        Identified stays with centroids
    poi_database : GeoDataFrame
        POI locations with geometry column
    poi_buffer : float
        Maximum distance in meters to match stay to POI
        
    Returns:
    --------
    mapped_stays : DataFrame
        Stays matched to POIs, with non-matched stays excluded
    """
    
    stays_geom = gpd.GeoDataFrame(
        stays,
        geometry=gpd.points_from_xy(stays['centroid_lon'], stays['centroid_lat'])
    )
    
    matched_stays = []
    for idx, stay in stays_geom.iterrows():
        # Find nearest POI
        distances = poi_database.geometry.distance(stay.geometry)
        nearest_poi_idx = distances.idxmin()
        nearest_distance = distances.iloc[nearest_poi_idx]
        
        # Keep stay if within buffer distance
        if nearest_distance <= poi_buffer / 111000:  # Convert meters to degrees
            stay_data = stay.drop('geometry').to_dict()
            stay_data['matched_poi_id'] = nearest_poi_idx
            stay_data['distance_to_poi_m'] = nearest_distance * 111000
            matched_stays.append(stay_data)
    
    return pd.DataFrame(matched_stays)


def filter_stays_temporal(stays, min_duration=15, max_duration=1440):
    """
    Applies temporal filters to identified stays
    
    Parameters:
    -----------
    stays : DataFrame
        Stays with duration information
    min_duration : float
        Minimum stay duration in minutes
    max_duration : float
        Maximum stay duration in minutes
        
    Returns:
    --------
    filtered_stays : DataFrame
        Stays within temporal thresholds
    """
    
    filtered_stays = stays[
        (stays['duration_minutes'] >= min_duration) &
        (stays['duration_minutes'] <= max_duration)
    ].copy()
    
    return filtered_stays


def extract_trips_from_stays(user_id, stays, gps_records):
    """
    Extracts trips as movements between consecutive filtered stays
    
    Parameters:
    -----------
    user_id : str
        User identifier
    stays : DataFrame
        Filtered stay locations sorted by time
    gps_records : DataFrame
        Original GPS records for validation
        
    Returns:
    --------
    trips : list
        List of dictionaries representing trips
        Each trip has: origin, destination, departure_time, arrival_time
    """
    
    stays = stays.sort_values('start_time').reset_index(drop=True)
    
    trips = []
    for i in range(len(stays) - 1):
        origin_stay = stays.iloc[i]
        destination_stay = stays.iloc[i + 1]
        
        trip = {
            'user_id': user_id,
            'origin_lon': origin_stay['centroid_lon'],
            'origin_lat': origin_stay['centroid_lat'],
            'destination_lon': destination_stay['centroid_lon'],
            'destination_lat': destination_stay['centroid_lat'],
            'departure_time': origin_stay['end_time'],
            'arrival_time': destination_stay['start_time'],
            'euclidean_distance_m': compute_distance(
                origin_stay['centroid_lat'], origin_stay['centroid_lon'],
                destination_stay['centroid_lat'], destination_stay['centroid_lon']
            )
        }
        trips.append(trip)
    
    return trips


def identify_home_location(user_trajectory, nighttime_start=21, nighttime_end=6, 
                          min_days_present=25, poi_database=None):
    """
    Identifies primary residence using multi-criteria approach
    
    Parameters:
    -----------
    user_trajectory : DataFrame
        User's GPS records with timestamps
    nighttime_start : int
        Hour when nighttime period begins (24-hour format)
    nighttime_end : int
        Hour when nighttime period ends
    min_days_present : int
        Minimum days at location in observation period
    poi_database : GeoDataFrame
        Residential POI database for validation
        
    Returns:
    --------
    home_location : dict
        Home coordinates and validation flags
    """
    
    # Extract timestamps and hour
    user_trajectory['timestamp'] = pd.to_datetime(user_trajectory['timestamp'])
    user_trajectory['hour'] = user_trajectory['timestamp'].dt.hour
    user_trajectory['date'] = user_trajectory['timestamp'].dt.date
    
    # Filter nighttime records (21:00 to 06:00)
    nighttime_mask = (
        (user_trajectory['hour'] >= nighttime_start) |
        (user_trajectory['hour'] < nighttime_end)
    )
    nighttime_records = user_trajectory[nighttime_mask]
    
    # Group by location (round for clustering)
    nighttime_records['lat_round'] = (nighttime_records['latitude'] * 1000).round() / 1000
    nighttime_records['lon_round'] = (nighttime_records['longitude'] * 1000).round() / 1000
    
    nighttime_records = nighttime_records.sort_values(['lat_round', 'lon_round', 'timestamp'])
    nighttime_records['time_diff'] = nighttime_records.groupby(
        ['lat_round', 'lon_round']
    )['timestamp'].diff().dt.total_seconds().fillna(0)
    nighttime_records.loc[nighttime_records['time_diff'] > 1800, 'time_diff'] = 0
    location_duration = nighttime_records.groupby(
        ['lat_round', 'lon_round']
    )['time_diff'].sum().reset_index()
    location_duration.columns = ['latitude', 'longitude', 'total_duration']

    # Find location with longest nighttime duration
    primary_home_idx = location_duration['total_duration'].idxmax()
    primary_home = location_duration.iloc[primary_home_idx]
    
    # Validate frequency
    nights_present = nighttime_records[
        (nighttime_records['lat_round'] == primary_home['latitude']) &
        (nighttime_records['lon_round'] == primary_home['longitude'])
    ]['date'].nunique()
    
    is_valid = nights_present >= min_days_present
    
    home_location = {
        'latitude': primary_home['latitude'],
        'longitude': primary_home['longitude'],
        'nights_present': nights_present,
        'is_valid': is_valid,
        'total_nighttime_minutes': primary_home['total_duration'] / 60
    }
    
    return home_location


def compute_distance(lat1, lon1, lat2, lon2):
    """
    Computes the Great Circle distance (Haversine) between two points.
    Far more accurate than Euclidean approximation for geospatial coordinates.
    
    Parameters:
    -----------
    lat1, lon1, lat2, lon2 : float or numpy.array
        Coordinates in decimal degrees
        
    Returns:
    --------
    distance_m : float or numpy.array
        Distance in meters
    """
    # Earth radius in meters
    R = 6371000.0
    
    # Convert decimal degrees to radians
    phi1, lambda1 = np.radians(lat1), np.radians(lon1)
    phi2, lambda2 = np.radians(lat2), np.radians(lon2)
    
    # Differences
    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1
    
    # Haversine formula
    a = np.sin(dphi / 2.0)**2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance_m = R * c
    
    return distance_m
