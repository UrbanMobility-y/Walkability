"""
Route reconstruction and route consistency validation.

AMap calls are isolated in ``RouteReconstructor``. All downstream validation and
metric calculations can be run with cached route geometries, which is how the
example pipeline and tests are executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point

from spatial_utils import line_point_distances_m, project_geometry_to_m


class RouteReconstructor:
    """Minimal AMap V5 route planning client."""

    def __init__(self, api_key: str, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout
        self.base_url = "https://restapi.amap.com/v5/direction"

    @staticmethod
    def decode_polyline(polyline_str: str) -> list[tuple[float, float]]:
        if not polyline_str:
            return []
        coords = []
        for token in polyline_str.split(";"):
            lon, lat = token.split(",")[:2]
            coords.append((float(lon), float(lat)))
        return coords

    def _request(self, endpoint: str, params: dict) -> dict | None:
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("info") != "OK" or not data.get("route", {}).get("paths"):
                return None
            return data
        except Exception:
            return None

    def get_route(self, origin: tuple[float, float], destination: tuple[float, float],
                  mode: Literal["walking", "driving"] = "walking") -> dict | None:
        params = {
            "key": self.api_key,
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
            "show_fields": "polyline,cost",
        }
        if mode == "walking":
            params.update({"alternative_route": 3, "isindoor": 0})
        elif mode == "driving":
            params.update({"strategy": 2, "cartype": 0})
        else:
            raise ValueError("mode must be 'walking' or 'driving'")
        data = self._request(mode, params)
        if data is None:
            return None
        path = data["route"]["paths"][0]
        coords = self.decode_polyline(path.get("polyline", ""))
        if len(coords) < 2:
            return None
        cost = path.get("cost", {})
        return {
            "geometry": LineString(coords),
            "distance_m": float(path.get("distance", 0.0)),
            "duration_sec": float(cost.get("duration", 0.0)),
            "steps": path.get("steps", []),
            "raw_response": data,
        }


def validate_route_geometry(gps_pings_lonlat: Iterable[tuple[float, float]], route_geom: LineString,
                            threshold_m: float = 100.0, ratio: float = 0.8) -> tuple[bool, float]:
    """Check whether enough intermediate pings fall within a metric buffer."""
    points = list(gps_pings_lonlat)
    if len(points) == 0 or route_geom is None or route_geom.is_empty:
        return False, 0.0
    distances = line_point_distances_m(points, route_geom)
    matched_ratio = float(np.mean(distances <= threshold_m))
    return matched_ratio >= ratio, matched_ratio


def route_quality_sensitivity(gps_pings_lonlat: Iterable[tuple[float, float]], route_geom: LineString,
                              buffers_m: Iterable[float] = (100, 80, 60, 40, 20), ratio: float = 0.8) -> pd.DataFrame:
    """Return retained/match ratios under alternative route consistency buffers."""
    points = list(gps_pings_lonlat)
    distances = line_point_distances_m(points, route_geom)
    rows = []
    for b in buffers_m:
        match_ratio = float(np.mean(distances <= b)) if len(distances) else 0.0
        rows.append({"buffer_m": b, "match_ratio": match_ratio, "retained": bool(match_ratio >= ratio)})
    return pd.DataFrame(rows)


def route_overlap_ratio(reference_route: LineString, candidate_route: LineString, ref_lat: float | None = None,
                        buffer_m: float = 5.0) -> float:
    """
    Approximate route overlap ratio after projecting both lines to metres.

    The numerator is the length of reference and candidate portions lying within
    a small mutual buffer; the denominator is the union length approximation.
    """
    if ref_lat is None:
        coords = list(reference_route.coords) + list(candidate_route.coords)
        ref_lat = float(np.mean([lat for _, lat in coords]))
    ref_m = project_geometry_to_m(reference_route, ref_lat)
    cand_m = project_geometry_to_m(candidate_route, ref_lat)
    if ref_m.length == 0 or cand_m.length == 0:
        return 0.0
    ref_shared = ref_m.intersection(cand_m.buffer(buffer_m)).length
    cand_shared = cand_m.intersection(ref_m.buffer(buffer_m)).length
    shared = 0.5 * (ref_shared + cand_shared)
    union = ref_m.length + cand_m.length - shared
    return float(np.clip(shared / union if union > 0 else 0.0, 0.0, 1.0))


def batch_validate_routes(routes_df: pd.DataFrame, pings_df: pd.DataFrame,
                          route_id_col: str = "trip_id", threshold_m: float = 100.0) -> pd.DataFrame:
    """Validate cached routes against intermediate pings."""
    rows = []
    for _, route in routes_df.iterrows():
        trip_id = route[route_id_col]
        pings = pings_df[pings_df[route_id_col] == trip_id][["longitude", "latitude"]].itertuples(index=False, name=None)
        ok, mr = validate_route_geometry(list(pings), route["geometry"], threshold_m=threshold_m)
        rows.append({route_id_col: trip_id, "valid_route": ok, "match_ratio": mr})
    return pd.DataFrame(rows)
