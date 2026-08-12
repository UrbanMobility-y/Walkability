"""
Mobile phone data processing.

This module implements the preprocessing, stay detection, OD extraction and
home location inference described in Supplementary Note 1.1. The public code is
designed for reproducible demonstration and validation. The raw telecom data
used in the paper cannot be redistributed, but the same functions operate on the
synthetic example data bundled with this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from spatial_utils import EARTH_RADIUS_M, haversine_distance_m, nearest_distance_to_points_m


REQUIRED_GPS_COLUMNS = {"user_id", "timestamp", "longitude", "latitude", "city_code"}


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str = "dataframe") -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def preprocess_raw_gps_data(gps_records: pd.DataFrame, min_monthly_records: int = 300):
    """
    Preprocess raw mobile phone pings.

    Steps: duplicate removal, filtering for residents observed in a single city, and sparse user
    filtering based on the monthly record count.
    """
    _require_columns(gps_records, REQUIRED_GPS_COLUMNS, "gps_records")
    gps_records = gps_records.copy()
    gps_records["timestamp"] = pd.to_datetime(gps_records["timestamp"])

    initial_users = int(gps_records["user_id"].nunique())
    initial_records = int(len(gps_records))

    gps_records = gps_records.drop_duplicates(subset=["user_id", "timestamp", "longitude", "latitude"])
    duplicates_removed = initial_records - len(gps_records)

    user_cities = gps_records.groupby("user_id")["city_code"].nunique()
    single_city_users = user_cities[user_cities == 1].index
    gps_records = gps_records[gps_records["user_id"].isin(single_city_users)].copy()

    gps_records["year_month"] = gps_records["timestamp"].dt.to_period("M")
    monthly_counts = gps_records.groupby(["user_id", "year_month"]).size().groupby("user_id").max()
    qualified_users = monthly_counts[monthly_counts >= min_monthly_records].index
    gps_records = gps_records[gps_records["user_id"].isin(qualified_users)].copy()

    final_users = int(gps_records["user_id"].nunique())
    final_records = int(len(gps_records))
    statistics = {
        "initial_users": initial_users,
        "initial_records": initial_records,
        "duplicates_removed": int(duplicates_removed),
        "users_after_preprocessing": final_users,
        "records_after_preprocessing": final_records,
        "retention_rate_users": final_users / initial_users if initial_users else np.nan,
        "retention_rate_records": final_records / initial_records if initial_records else np.nan,
    }
    return gps_records.drop(columns=["year_month"], errors="ignore"), statistics


def identify_stays_dbscan(user_trajectory: pd.DataFrame, eps_m: float = 50.0, min_pts: int = 10) -> pd.DataFrame:
    """
    Identify recurrent spatial activity clusters using haversine DBSCAN.

    DBSCAN is applied in spherical coordinates using ``metric='haversine'`` and
    ``eps_m / Earth_radius``. This avoids the flat ``meters / 111000`` conversion
    and the associated longitude compression error at Chinese latitudes.

    Note on the 5-minute stay threshold used in the manuscript: DBSCAN detects
    monthly recurrent spatial clusters. Visit level dwell times are then derived
    from temporally consecutive pings within each cluster and filtered by
    ``filter_stays_temporal``. Thus ``min_pts`` controls spatial cluster
    reliability and does not require 10 pings within a single 5-minute visit.
    """
    _require_columns(user_trajectory, ["timestamp", "longitude", "latitude"], "user_trajectory")
    if user_trajectory.empty:
        return pd.DataFrame(columns=["cluster", "centroid_lon", "centroid_lat", "num_pings", "start_time", "end_time", "duration_minutes"])

    traj = user_trajectory.copy().sort_values("timestamp")
    traj["timestamp"] = pd.to_datetime(traj["timestamp"])
    coords_rad = np.radians(traj[["latitude", "longitude"]].to_numpy(dtype=float))
    clustering = DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=min_pts, metric="haversine")
    labels = clustering.fit_predict(coords_rad)
    traj["cluster"] = labels

    # A DBSCAN label identifies a recurrent monthly activity location, not one
    # continuous stay. Reconstruct separate visits from consecutive label runs
    # before applying the duration threshold to each visit.
    traj["visit_run"] = traj["cluster"].ne(traj["cluster"].shift()).cumsum()
    stays = []
    for (_, cluster_id), group in traj[traj["cluster"] != -1].groupby(["visit_run", "cluster"]):
        group = group.sort_values("timestamp")
        stays.append({
            "cluster": int(cluster_id),
            "centroid_lon": float(group["longitude"].mean()),
            "centroid_lat": float(group["latitude"].mean()),
            "num_pings": int(len(group)),
            "start_time": group["timestamp"].min(),
            "end_time": group["timestamp"].max(),
            "duration_minutes": float((group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 60.0),
        })
    return pd.DataFrame(stays).sort_values("start_time").reset_index(drop=True) if stays else pd.DataFrame(
        columns=["cluster", "centroid_lon", "centroid_lat", "num_pings", "start_time", "end_time", "duration_minutes"]
    )


def map_stays_to_poi(stays: pd.DataFrame, poi_database: pd.DataFrame, poi_buffer_m: float = 100.0) -> pd.DataFrame:
    """Map stays to the nearest POI using haversine distance in metres."""
    if stays.empty:
        return stays.copy()
    _require_columns(stays, ["centroid_lon", "centroid_lat"], "stays")
    _require_columns(poi_database, ["longitude", "latitude"], "poi_database")

    poi_lons = poi_database["longitude"].to_numpy(dtype=float)
    poi_lats = poi_database["latitude"].to_numpy(dtype=float)
    records = []
    for _, stay in stays.iterrows():
        nearest_pos, dist_m = nearest_distance_to_points_m(stay["centroid_lon"], stay["centroid_lat"], poi_lons, poi_lats)
        if dist_m <= poi_buffer_m:
            item = stay.to_dict()
            item["matched_poi_id"] = poi_database.index[nearest_pos]
            item["distance_to_poi_m"] = dist_m
            if "category" in poi_database.columns:
                item["poi_category"] = poi_database.iloc[nearest_pos]["category"]
            records.append(item)
    return pd.DataFrame(records)


def filter_stays_temporal(stays: pd.DataFrame, min_duration: float = 5.0, max_duration: float = 1440.0) -> pd.DataFrame:
    """Apply the manuscript baseline 5-minute stay duration threshold."""
    if stays.empty:
        return stays.copy()
    return stays[(stays["duration_minutes"] >= min_duration) & (stays["duration_minutes"] <= max_duration)].copy()


def extract_trips_from_stays(user_id: str, stays: pd.DataFrame) -> list[dict]:
    """Extract OD trips between consecutive filtered stays."""
    if stays.empty or len(stays) < 2:
        return []
    stays = stays.sort_values("start_time").reset_index(drop=True)
    trips: list[dict] = []
    for i in range(len(stays) - 1):
        o = stays.iloc[i]
        d = stays.iloc[i + 1]
        trips.append({
            "user_id": user_id,
            "origin_lon": float(o["centroid_lon"]),
            "origin_lat": float(o["centroid_lat"]),
            "destination_lon": float(d["centroid_lon"]),
            "destination_lat": float(d["centroid_lat"]),
            "departure_time": o["end_time"],
            "arrival_time": d["start_time"],
            "euclidean_distance_m": float(haversine_distance_m(o["centroid_lat"], o["centroid_lon"], d["centroid_lat"], d["centroid_lon"])),
        })
    return trips


def _nighttime_overlap_rows(start_time: pd.Timestamp, end_time: pd.Timestamp,
                            nighttime_start: int, nighttime_end: int) -> list[dict]:
    """Split a stay into its overlaps with evening to morning night windows."""
    if end_time < start_time:
        raise ValueError("A stay end_time cannot precede its start_time")
    rows: list[dict] = []
    first_evening = (start_time.normalize() - pd.Timedelta(days=1))
    last_evening = end_time.normalize()
    for evening in pd.date_range(first_evening, last_evening, freq="D"):
        window_start = evening + pd.Timedelta(hours=nighttime_start)
        window_end = evening + pd.Timedelta(days=1, hours=nighttime_end)
        overlap_start = max(start_time, window_start)
        overlap_end = min(end_time, window_end)
        if overlap_end > overlap_start:
            weekend_minutes = 0.0
            for day in pd.date_range(overlap_start.normalize(), overlap_end.normalize(), freq="D"):
                if day.dayofweek >= 5:
                    day_start = day
                    day_end = day + pd.Timedelta(days=1)
                    weekend_start = max(overlap_start, day_start)
                    weekend_end = min(overlap_end, day_end)
                    if weekend_end > weekend_start:
                        weekend_minutes += (weekend_end - weekend_start).total_seconds() / 60.0
            rows.append({
                "night_date": evening.date(),
                "nighttime_duration_minutes": (overlap_end - overlap_start).total_seconds() / 60.0,
                "weekend_duration_minutes": weekend_minutes,
            })
    return rows


def identify_home_location(
    identified_stays: pd.DataFrame,
    nighttime_start: int = 21,
    nighttime_end: int = 6,
    min_days_present: int = 25,
    residential_categories: Iterable[str] = ("residential",),
    require_residential_poi: bool = True,
    weekend_validator: Optional[Callable[[pd.Series], bool]] = None,
) -> dict:
    """Identify a primary residence from already identified and POI mapped stays.

    Candidate locations are evaluated by cumulative overlap with the 21:00 to
    06:00 window, presence on at least 25 distinct nights, weekend duration and
    residential POI status. If multiple candidates pass, the one with the
    greatest cumulative nighttime duration is selected.

    The paper does not report a numerical weekend threshold. The public method
    therefore reports weekend duration for every candidate and accepts an
    optional ``weekend_validator`` for applying the production study's locally
    specified rule without inventing an undocumented cutoff.

    ``identified_stays`` must contain visit level rows returned by
    ``identify_stays_dbscan``. Residential status can be supplied through a
    Boolean ``is_residential_poi`` column or a ``poi_category`` column produced
    by ``map_stays_to_poi``.
    """
    required = ["cluster", "centroid_lon", "centroid_lat", "start_time", "end_time"]
    _require_columns(identified_stays, required, "identified_stays")
    if identified_stays.empty:
        return {"is_valid": False, "reason": "no identified stays"}
    if not 0 <= nighttime_start <= 23 or not 0 <= nighttime_end <= 23:
        raise ValueError("nighttime_start and nighttime_end must be hours from 0 to 23")
    if nighttime_start <= nighttime_end:
        raise ValueError("This implementation expects a nighttime window that crosses midnight")

    stays = identified_stays.copy()
    stays["start_time"] = pd.to_datetime(stays["start_time"])
    stays["end_time"] = pd.to_datetime(stays["end_time"])
    residential_values = {str(value).strip().casefold() for value in residential_categories}
    candidate_rows: list[dict] = []

    for cluster_id, visits in stays.groupby("cluster", sort=False):
        night_rows: list[dict] = []
        for _, visit in visits.iterrows():
            night_rows.extend(_nighttime_overlap_rows(
                visit["start_time"], visit["end_time"], nighttime_start, nighttime_end
            ))
        if not night_rows:
            continue
        overlaps = pd.DataFrame(night_rows)
        by_night = overlaps.groupby("night_date", as_index=False).agg(
            nighttime_duration_minutes=("nighttime_duration_minutes", "sum"),
            weekend_duration_minutes=("weekend_duration_minutes", "sum"),
        )
        if "is_residential_poi" in visits.columns:
            is_residential = bool(visits["is_residential_poi"].fillna(False).any())
        elif "poi_category" in visits.columns:
            categories = visits["poi_category"].dropna().astype(str).str.strip().str.casefold()
            is_residential = bool(categories.isin(residential_values).any())
        else:
            if require_residential_poi:
                raise ValueError(
                    "identified_stays needs poi_category or is_residential_poi when "
                    "require_residential_poi is True"
                )
            is_residential = False

        candidate_rows.append({
            "cluster": cluster_id,
            "longitude": float(visits["centroid_lon"].mean()),
            "latitude": float(visits["centroid_lat"].mean()),
            "nights_present": int(by_night["night_date"].nunique()),
            "nighttime_duration_minutes": float(by_night["nighttime_duration_minutes"].sum()),
            "weekend_nights_present": int(
                by_night.loc[by_night["weekend_duration_minutes"] > 0, "night_date"].nunique()
            ),
            "weekend_nighttime_duration_minutes": float(by_night["weekend_duration_minutes"].sum()),
            "is_residential_poi": is_residential,
        })

    if not candidate_rows:
        return {"is_valid": False, "reason": "no nighttime stays"}
    candidates = pd.DataFrame(candidate_rows)
    candidates["passes_frequency_check"] = candidates["nights_present"] >= min_days_present
    candidates["passes_residential_check"] = (
        candidates["is_residential_poi"] if require_residential_poi else True
    )
    candidates["passes_weekend_check"] = (
        candidates.apply(weekend_validator, axis=1) if weekend_validator is not None else True
    )
    eligible = candidates[
        candidates["passes_frequency_check"]
        & candidates["passes_residential_check"]
        & candidates["passes_weekend_check"]
    ].copy()
    if eligible.empty:
        return {
            "is_valid": False,
            "reason": "no candidate passed all configured checks",
            "candidates": candidates.to_dict(orient="records"),
        }
    best = eligible.sort_values(
        ["nighttime_duration_minutes", "weekend_nighttime_duration_minutes"],
        ascending=False,
    ).iloc[0]
    result = best.to_dict()
    result.update({
        "is_valid": True,
        "weekend_check": "custom validator" if weekend_validator is not None else "reported, no published cutoff",
        "candidates": candidates.to_dict(orient="records"),
    })
    return result


def process_user_trajectory(user_id: str, user_trajectory: pd.DataFrame, poi_database: pd.DataFrame,
                            eps_m: float = 50.0, min_pts: int = 10, min_duration: float = 5.0,
                            poi_buffer_m: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete stay and trip extraction workflow for a single user."""
    stays = identify_stays_dbscan(user_trajectory, eps_m=eps_m, min_pts=min_pts)
    mapped = map_stays_to_poi(stays, poi_database, poi_buffer_m=poi_buffer_m)
    filtered = filter_stays_temporal(mapped, min_duration=min_duration)
    trips = pd.DataFrame(extract_trips_from_stays(user_id, filtered))
    return filtered, trips
