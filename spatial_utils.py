"""
Spatial utility functions used across the walkability workflow.

The functions in this file avoid the meter to degree approximation that can
produce errors that depend on latitude in east west distances. They use haversine
distances for point to point calculations and a local equirectangular projection
for geometry operations in small areas.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import transform

EARTH_RADIUS_M = 6_371_000.0
LAT_M_PER_DEG = 111_320.0


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Great circle distance in metres for scalars or NumPy/Pandas arrays."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def project_lonlat_to_m(lon, lat, ref_lat: float | None = None):
    """Project lon/lat to local metres using a reference latitude."""
    if ref_lat is None:
        ref_lat = float(np.nanmean(lat)) if np.ndim(lat) else float(lat)
    lon_factor = LAT_M_PER_DEG * np.cos(np.radians(ref_lat))
    lat_factor = LAT_M_PER_DEG
    return np.asarray(lon) * lon_factor, np.asarray(lat) * lat_factor


def make_local_projector(ref_lat: float):
    """Return a local projection function compatible with Shapely."""
    lon_factor = LAT_M_PER_DEG * np.cos(np.radians(ref_lat))
    lat_factor = LAT_M_PER_DEG

    def _project(x, y, z=None):
        x_m = np.asarray(x) * lon_factor
        y_m = np.asarray(y) * lat_factor
        if z is None:
            return x_m, y_m
        return x_m, y_m, z

    return _project


def project_geometry_to_m(geom, ref_lat: float):
    """Project a Shapely geometry from lon/lat coordinates to local metres."""
    return transform(make_local_projector(ref_lat), geom)


def nearest_distance_to_points_m(lon: float, lat: float, candidate_lons: Iterable[float], candidate_lats: Iterable[float]):
    """Return nearest candidate index and haversine distance in metres."""
    d = haversine_distance_m(lat, lon, np.asarray(candidate_lats), np.asarray(candidate_lons))
    idx = int(np.nanargmin(d))
    return idx, float(d[idx])


def line_point_distances_m(points_lonlat: Iterable[Tuple[float, float]], route_geom: LineString):
    """Perpendicular distances in metres from lon/lat points to a route LineString."""
    points = list(points_lonlat)
    if not points:
        return np.array([])
    ref_lat = float(np.mean([p[1] for p in points]))
    route_m = project_geometry_to_m(route_geom, ref_lat)
    projector = make_local_projector(ref_lat)
    distances = []
    for lon, lat in points:
        x, y = projector(lon, lat)
        distances.append(Point(float(x), float(y)).distance(route_m))
    return np.asarray(distances, dtype=float)
