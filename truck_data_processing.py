"""
Heavy truck GPS processing.

The truck component captures regulated heavy freight vehicles covered by the
China Road Freight Supervision and Service Platform (gross weight >= 12 tons).
It does not represent all last mile urban logistics such as light vans, courier
motorcycles, cargo e-bikes or three wheeled delivery vehicles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from shapely.geometry import Point

from spatial_utils import haversine_distance_m


def preprocess_truck_gps(gps_records: pd.DataFrame, max_consecutive_gap: float = 60.0,
                         max_speed_change: float = 50.0, spatial_outlier_threshold_m: float = 500.0) -> pd.DataFrame:
    """Clean raw heavy truck GPS trajectories."""
    required = {"truck_id", "timestamp", "longitude", "latitude", "speed_kmh"}
    missing = required - set(gps_records.columns)
    if missing:
        raise ValueError(f"gps_records missing columns: {sorted(missing)}")
    df = gps_records.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["truck_id", "timestamp"])

    df["time_gap"] = df.groupby("truck_id")["timestamp"].diff().dt.total_seconds()
    loss_rate = df.groupby("truck_id")["time_gap"].apply(lambda s: (s > max_consecutive_gap).mean())
    valid_ids = loss_rate[loss_rate <= 0.05].index
    df = df[df["truck_id"].isin(valid_ids)].copy()
    df = df.drop_duplicates(subset=["truck_id", "timestamp", "longitude", "latitude"])

    g = df.groupby("truck_id")
    prev_lat, prev_lon = g["latitude"].shift(1), g["longitude"].shift(1)
    next_lat, next_lon = g["latitude"].shift(-1), g["longitude"].shift(-1)
    dist_prev = haversine_distance_m(df["latitude"], df["longitude"], prev_lat, prev_lon)
    dist_next = haversine_distance_m(df["latitude"], df["longitude"], next_lat, next_lon)
    is_outlier = (dist_prev > spatial_outlier_threshold_m) & (dist_next > spatial_outlier_threshold_m)
    df = df[~pd.Series(is_outlier, index=df.index).fillna(False)].copy()

    speed_diff = df.groupby("truck_id")["speed_kmh"].diff().abs()
    df = df[(speed_diff.isna()) | (speed_diff <= max_speed_change)].copy()
    return df.reset_index(drop=True)


def estimate_stop_speed_threshold_gmm(speeds_kmh, min_threshold: float = 0.5, max_threshold: float = 10.0) -> float:
    """
    Estimate a stop/move speed threshold using a Gaussian mixture with two components.

    This lightweight implementation uses expectation maximization directly in
    NumPy/Scipy rather than relying on external mixture model objects. The
    threshold is the saddle point of the fitted mixture density between the
    means of the lower and higher speed components.
    """
    speeds = np.asarray(speeds_kmh, dtype=float)
    speeds = speeds[np.isfinite(speeds)]
    speeds = speeds[(speeds >= 0) & (speeds <= 120)]
    if len(speeds) < 30 or np.nanstd(speeds) < 0.1:
        return 3.0

    # Robust initialization from quantiles.
    means = np.array([np.percentile(speeds, 15), np.percentile(speeds, 75)], dtype=float)
    variances = np.array([max(np.var(speeds[speeds <= np.median(speeds)]), 0.25),
                          max(np.var(speeds[speeds > np.median(speeds)]), 1.0)], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)

    x = speeds[:, None]
    for _ in range(100):
        densities = np.column_stack([
            weights[k] * norm.pdf(speeds, means[k], np.sqrt(max(variances[k], 1e-6)))
            for k in range(2)
        ])
        denom = densities.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1e-12
        resp = densities / denom
        Nk = resp.sum(axis=0)
        new_weights = Nk / len(speeds)
        new_means = (resp * speeds[:, None]).sum(axis=0) / np.maximum(Nk, 1e-12)
        new_vars = (resp * (speeds[:, None] - new_means) ** 2).sum(axis=0) / np.maximum(Nk, 1e-12)
        if np.max(np.abs(new_means - means)) < 1e-4:
            means, variances, weights = new_means, np.maximum(new_vars, 1e-6), new_weights
            break
        means, variances, weights = new_means, np.maximum(new_vars, 1e-6), new_weights

    order = np.argsort(means)
    low, high = order[0], order[1]
    m_low, m_high = means[low], means[high]
    if not np.isfinite(m_low + m_high) or m_high <= m_low:
        return 3.0

    grid_min = max(min_threshold, float(m_low))
    grid_max = min(max_threshold, float(m_high))
    if grid_max <= grid_min:
        return float(np.clip((m_low + m_high) / 2.0, min_threshold, max_threshold))
    grid = np.linspace(grid_min, grid_max, 500)
    density = np.zeros_like(grid)
    for k in range(2):
        density += weights[k] * norm.pdf(grid, means[k], np.sqrt(max(variances[k], 1e-6)))
    threshold = float(grid[np.argmin(density)])
    return float(np.clip(threshold, min_threshold, max_threshold))

def identify_truck_stops_adaptive(truck_trajectory: pd.DataFrame, method: str = "gmm") -> pd.DataFrame:
    """Identify stationary periods in a heavy truck trajectory."""
    if truck_trajectory.empty:
        return pd.DataFrame()
    df = truck_trajectory.copy().sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if method != "gmm":
        raise ValueError("Only method='gmm' is provided in the reproducible implementation.")
    threshold = estimate_stop_speed_threshold_gmm(df["speed_kmh"])
    df["is_stop"] = df["speed_kmh"] < threshold
    df["stop_group"] = (df["is_stop"] != df["is_stop"].shift()).cumsum()

    records = []
    for _, group in df[df["is_stop"]].groupby("stop_group"):
        start = group["timestamp"].min()
        end = group["timestamp"].max()
        records.append({
            "centroid_lat": float(group["latitude"].mean()),
            "centroid_lon": float(group["longitude"].mean()),
            "start_time": start,
            "end_time": end,
            "duration_minutes": float((end - start).total_seconds() / 60.0),
            "speed_threshold_kmh": threshold,
        })
    return pd.DataFrame(records)


def identify_truck_trip_ods_recursive(all_stops: pd.DataFrame, gps_trajectory: pd.DataFrame,
                                      long_duration_threshold: float = 240.0,
                                      short_duration_threshold: float = 20.0,
                                      max_circuity: float = 2.0,
                                      freight_pois: pd.DataFrame | None = None) -> pd.DataFrame:
    """Identify truck trip ODs through hierarchical stop duration and circuity checks."""
    if all_stops.empty:
        return pd.DataFrame()
    stops = all_stops.sort_values("start_time").reset_index(drop=True)
    gps = gps_trajectory.copy().sort_values("timestamp")
    gps["timestamp"] = pd.to_datetime(gps["timestamp"])
    gps_times = gps["timestamp"].to_numpy()

    step_dist = haversine_distance_m(gps["latitude"], gps["longitude"], gps["latitude"].shift(1), gps["longitude"].shift(1))
    gps["cum_dist_m"] = pd.Series(step_dist).fillna(0).cumsum().to_numpy()
    cum_dist = gps["cum_dist_m"].to_numpy()

    def valid_freight_location(stop) -> bool:
        if freight_pois is None or freight_pois.empty:
            return True
        d = haversine_distance_m(stop["centroid_lat"], stop["centroid_lon"], freight_pois["latitude"], freight_pois["longitude"])
        if "category" in freight_pois.columns:
            cats = freight_pois.loc[np.asarray(d) <= 200, "category"].astype(str).str.lower()
            return cats.str.contains("warehouse|factory|logistics|industrial").any()
        return bool((np.asarray(d) <= 200).any())

    def path_distance_between(t0, t1) -> float:
        i0 = int(np.searchsorted(gps_times, np.datetime64(t0), side="left"))
        i1 = int(np.searchsorted(gps_times, np.datetime64(t1), side="right") - 1)
        i0 = max(0, min(i0, len(cum_dist) - 1))
        i1 = max(0, min(i1, len(cum_dist) - 1))
        return float(max(cum_dist[i1] - cum_dist[i0], 0.0))

    def recurse(origin, dest, candidates):
        t0, t1 = origin["end_time"], dest["start_time"]
        path_len = path_distance_between(t0, t1)
        euc = float(haversine_distance_m(origin["centroid_lat"], origin["centroid_lon"], dest["centroid_lat"], dest["centroid_lon"]))
        circuity = path_len / euc if euc > 100 else 1.0
        if circuity <= max_circuity or candidates.empty:
            if valid_freight_location(origin) and valid_freight_location(dest):
                return [{
                    "origin_lat": origin["centroid_lat"], "origin_lon": origin["centroid_lon"],
                    "dest_lat": dest["centroid_lat"], "dest_lon": dest["centroid_lon"],
                    "start_time": t0, "end_time": t1, "distance_m": path_len, "circuity": circuity,
                }]
            return []
        pivot = candidates.loc[candidates["duration_minutes"].idxmax()]
        left = candidates[candidates["end_time"] <= pivot["start_time"]]
        right = candidates[candidates["start_time"] >= pivot["end_time"]]
        return recurse(origin, pivot, left) + recurse(pivot, dest, right)

    primary = stops[stops["duration_minutes"] >= long_duration_threshold]
    if len(primary) < 2:
        return pd.DataFrame()
    trips = []
    for i in range(len(primary) - 1):
        o = primary.iloc[i]
        d = primary.iloc[i + 1]
        mask = (
            (stops["start_time"] >= o["end_time"]) &
            (stops["end_time"] <= d["start_time"]) &
            (stops["duration_minutes"] >= short_duration_threshold) &
            (stops["duration_minutes"] < long_duration_threshold)
        )
        trips.extend(recurse(o, d, stops[mask]))
    return pd.DataFrame(trips)


def filter_on_major_roads(trip_ods: pd.DataFrame, major_road_points: pd.DataFrame, threshold_m: float = 20.0) -> pd.DataFrame:
    """Remove ODs that fall directly on major roads using point samples along roads."""
    if trip_ods.empty or major_road_points.empty:
        return trip_ods.copy()
    keep = []
    for _, row in trip_ods.iterrows():
        d_o = np.min(haversine_distance_m(row["origin_lat"], row["origin_lon"], major_road_points["latitude"], major_road_points["longitude"]))
        d_d = np.min(haversine_distance_m(row["dest_lat"], row["dest_lon"], major_road_points["latitude"], major_road_points["longitude"]))
        keep.append((d_o > threshold_m) and (d_d > threshold_m))
    return trip_ods[pd.Series(keep, index=trip_ods.index)].copy()
