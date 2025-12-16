"""
Transport Mode Inference Module (Final Corrected)
Infers transport modes (active, private, public) using XGBoost.

"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import cKDTree
import requests


class AMapRoutingAPI:
    """Interface to AutoNavi (AMap) routing API."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://restapi.amap.com/v3/direction"
    
    def _safe_request(self, endpoint, params):
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=2)
            if response.status_code == 200:
                result = response.json()
                if result['status'] == '1' and result['route']['paths']:
                    return int(result['route']['paths'][0]['duration'])
                elif endpoint == 'transit' and result['status'] == '1' and result['route']['transits']:
                    return int(result['route']['transits'][0]['duration'])
        except Exception:
            return None
        return None

    def get_travel_times(self, o_lon, o_lat, d_lon, d_lat):
        """Returns tuple of (walk_time, drive_time, transit_time)."""
        base_params = {
            'key': self.api_key, 
            'origin': f"{o_lon},{o_lat}", 
            'destination': f"{d_lon},{d_lat}"
        }
        
        walk_t = self._safe_request('walking', base_params)
        drive_t = self._safe_request('driving', {**base_params, 'strategy': 2})
        # Note: 'city' param is technically required for transit but omitted for brevity
        transit_t = self._safe_request('transit', {**base_params, 'city': '010', 'strategy': 0})
        
        return walk_t, drive_t, transit_t


# ==============================================================================
# 2. Feature Engineering
# ==============================================================================

def compute_trip_features(trips_df, amap_api, poi_database, city_center):
    """
    Computes features including relative time ratios and spatial proximity.
    Uses Local Projection for accurate KDTree queries.
    """
    
    # Project (lon, lat) -> (x_meters, y_meters) to make KDTree valid
    # Use mean latitude of the city center for projection factor
    mean_lat = city_center[1]
    lat_factor = 111000
    lon_factor = 111000 * np.cos(np.radians(mean_lat))
    
    def project(lon, lat):
        return np.column_stack([lon * lon_factor, lat * lat_factor])

    # Prepare Transit POIs
    transit_pois = poi_database[
        poi_database['category'].isin(['bus_station', 'metro_station'])
    ]
    
    # Build Tree on PROJECTED coordinates
    station_coords_proj = project(transit_pois['longitude'].values, 
                                  transit_pois['latitude'].values)
    station_tree = cKDTree(station_coords_proj)

    features = []
    
    # Distance utility (Euclidean on projected coords is accurate enough locally)
    def dist_metric(x1, y1, x2, y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    # City center projected
    cx, cy = city_center[0] * lon_factor, city_center[1] * lat_factor

    for idx, trip in trips_df.iterrows():
        olat, olon = trip['origin_lat'], trip['origin_lon']
        dlat, dlon = trip['destination_lat'], trip['destination_lon']
        
        # Project Trip OD
        ox, oy = olon * lon_factor, olat * lat_factor
        dx, dy = dlon * lon_factor, dlat * lat_factor
        
        # A. Spatial Features
        dist_od = dist_metric(ox, oy, dx, dy)
        dist_o_center = dist_metric(ox, oy, cx, cy)
        dist_d_center = dist_metric(dx, dy, cx, cy)
        
        # Fast nearest station lookup (using projected coords)
        dist_o_station, _ = station_tree.query([ox, oy])
        dist_d_station, _ = station_tree.query([dx, dy])
        
        # B. Temporal Features
        dt = pd.to_datetime(trip['departure_time'])
        hour = dt.hour
        is_peak = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        
        # C. API Travel Times
        w_time, d_time, t_time = amap_api.get_travel_times(olon, olat, dlon, dlat)
        
        if None in [w_time, d_time, t_time]:
            continue 
            
        # D. Ratio Features (Relative Advantage)
        # Adding small epsilon to avoid division by zero
        ratio_transit_drive = t_time / (d_time + 1.0)
        ratio_walk_drive = w_time / (d_time + 1.0)
        
        features.append({
            'dist_OD': dist_od,
            'dist_O_station': dist_o_station,
            'dist_D_station': dist_d_station,
            'dist_O_center': dist_o_center,
            'dist_D_center': dist_d_center,
            'hour': hour,
            'is_peak': is_peak,
            'time_walk': w_time,
            'time_drive': d_time,
            'time_transit': t_time,
            'ratio_transit_drive': ratio_transit_drive,
            'ratio_walk_drive': ratio_walk_drive
        })
        
    return pd.DataFrame(features)


class TransportModeClassifier:
    """XGBoost Classifier for mode inference."""
    
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=3,
            n_jobs=-1,
            random_state=42
        )
        self.le = LabelEncoder()
        
    def train(self, X, y):
        """Trains with Stratified CV."""
        y_enc = self.le.fit_transform(y)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y_enc, cv=skf, scoring='accuracy')
        self.model.fit(X, y_enc)
        return scores.mean()

    def predict(self, X):
        """Returns labels and probabilities."""
        if not hasattr(self.model, "feature_importances_"):
            raise Exception("Model not trained yet.")
            
        probs = self.model.predict_proba(X)
        preds_idx = np.argmax(probs, axis=1)
        preds_label = self.le.inverse_transform(preds_idx)
        
        # Robustly map probabilities to classes based on Encoder order
        # LabelEncoder sorts classes alphabetically: 
        # e.g., ['active', 'private', 'public'] -> [0, 1, 2]
        class_order = self.le.classes_
        
        result = pd.DataFrame({'pred_mode': preds_label})
        for i, cls in enumerate(class_order):
            result[f'prob_{cls}'] = probs[:, i]
            
        return result