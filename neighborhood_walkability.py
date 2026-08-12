"""Residential catchments and neighborhood level walkability.

The manuscript uses two distinct neighborhood representations:

* fixed 0.005-degree cells for the city/urban form analysis; and
* home centered, circuity adjusted 15-minute catchments for resident level
  analysis and all policy scenarios.

This module implements the second representation and the common
length weighted aggregation. Coordinates supplied to the geometric helpers
must be in a projected CRS whose unit is metres.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point


def circuity_adjusted_radius_m(
    mean_circuity: float,
    walk_minutes: float = 15.0,
    walk_speed_mps: float = 1.0,
) -> float:
    """Return ``network budget / mean circuity`` as used in the manuscript."""
    mean_circuity = float(mean_circuity)
    if not np.isfinite(mean_circuity) or mean_circuity <= 0:
        raise ValueError("mean_circuity must be a positive finite value")
    if walk_minutes <= 0 or walk_speed_mps <= 0:
        raise ValueError("walk_minutes and walk_speed_mps must be positive")
    return float(walk_minutes * 60.0 * walk_speed_mps / mean_circuity)


def estimate_mean_circuity(
    network_lengths_m: Iterable[float],
    euclidean_lengths_m: Iterable[float],
) -> float:
    """Estimate the city mean of network/Euclidean ratios for individual trips."""
    network = np.asarray(list(network_lengths_m), dtype=float)
    euclidean = np.asarray(list(euclidean_lengths_m), dtype=float)
    if network.shape != euclidean.shape:
        raise ValueError("network and Euclidean length arrays must have the same shape")
    valid = np.isfinite(network) & np.isfinite(euclidean) & (network > 0) & (euclidean > 0)
    if not valid.any():
        raise ValueError("no valid positive trip lengths were supplied")
    return float(np.mean(network[valid] / euclidean[valid]))


def _iter_edges(graph: nx.Graph):
    if graph.is_multigraph():
        yield from graph.edges(keys=True, data=True)
    else:
        for u, v, data in graph.edges(data=True):
            yield u, v, None, data


def _edge_geometry(graph: nx.Graph, u, v, data: Mapping) -> LineString:
    geometry = data.get("geometry")
    if geometry is not None:
        if not isinstance(geometry, LineString):
            raise TypeError("edge geometry must be a Shapely LineString")
        return geometry
    try:
        return LineString(
            [
                (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
                (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
            ]
        )
    except KeyError as exc:
        raise ValueError(
            "catchment inference requires edge geometry or metric node x/y coordinates"
        ) from exc


def _home_xy(graph: nx.Graph, trip: Mapping) -> tuple[float, float]:
    if "home_x" in trip and "home_y" in trip:
        return float(trip["home_x"]), float(trip["home_y"])
    home_node = trip.get("home_node", trip.get("origin_node"))
    if home_node in graph and "x" in graph.nodes[home_node] and "y" in graph.nodes[home_node]:
        return float(graph.nodes[home_node]["x"]), float(graph.nodes[home_node]["y"])
    raise ValueError("trip needs home_x/home_y or a home_node with metric x/y coordinates")


def catchment_edge_keys(
    graph: nx.Graph,
    trip: Mapping,
    default_radius_m: float | None = None,
) -> set[tuple]:
    """Resolve the street segments in one resident's 15-minute catchment.

    For production workflows backed by a database, callers may provide an explicit
    ``catchment_edges`` iterable containing edge IDs or ``(u, v[, key])``
    tuples. For the public demonstration, the set can be inferred from metric
    graph geometries and ``catchment_radius_m``.

    Policy perturbations use only the segments returned for the defined
    residential catchment.
    """
    explicit = trip.get("catchment_edges")
    if explicit is not None and not (isinstance(explicit, float) and np.isnan(explicit)):
        requested_ids: set = set()
        requested_tuples: set[tuple] = set()
        for item in explicit:
            if isinstance(item, (tuple, list)):
                requested_tuples.add(tuple(item))
            else:
                requested_ids.add(item)
        selected: set[tuple] = set()
        for u, v, key, data in _iter_edges(graph):
            identity = (u, v, key) if key is not None else (u, v)
            reverse = (v, u, key) if key is not None else (v, u)
            if identity in requested_tuples or reverse in requested_tuples or data.get("edge_id") in requested_ids:
                selected.add((u, v, key))
        return selected

    radius = trip.get("catchment_radius_m", default_radius_m)
    if radius is None or not np.isfinite(float(radius)) or float(radius) <= 0:
        raise ValueError(
            "policy trips require catchment_edges or a positive catchment_radius_m"
        )
    home = Point(_home_xy(graph, trip))
    boundary = home.buffer(float(radius))
    return {
        (u, v, key)
        for u, v, key, data in _iter_edges(graph)
        if _edge_geometry(graph, u, v, data).intersects(boundary)
    }


def clipped_catchment_lengths(
    graph: nx.Graph,
    home_x: float,
    home_y: float,
    radius_m: float,
) -> dict[tuple, float]:
    """Return the length of each edge portion lying inside a circular catchment."""
    boundary = Point(float(home_x), float(home_y)).buffer(float(radius_m))
    lengths: dict[tuple, float] = {}
    for u, v, key, data in _iter_edges(graph):
        clipped = _edge_geometry(graph, u, v, data).intersection(boundary)
        if not clipped.is_empty and clipped.length > 0:
            lengths[(u, v, key)] = float(clipped.length)
    return lengths


def length_weighted_mean(values: Iterable[float], lengths_m: Iterable[float]) -> float:
    """Compute a weighted mean using finite values and positive lengths."""
    values = np.asarray(list(values), dtype=float)
    lengths = np.asarray(list(lengths_m), dtype=float)
    if values.shape != lengths.shape:
        raise ValueError("values and lengths must have the same shape")
    valid = np.isfinite(values) & np.isfinite(lengths) & (lengths > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=lengths[valid]))


def aggregate_fixed_grid_cells(
    clipped_segments: pd.DataFrame,
    cell_col: str = "cell_id",
    length_col: str = "length_in_cell_m",
    swi_col: str = "SWI",
    dwi_col: str = "DWI",
) -> pd.DataFrame:
    """Aggregate clipped street pieces to fixed 0.005-degree analysis cells.

    This function performs the manuscript's length weighting after spatial
    overlay in GeoPandas or database tooling, with every row representing a
    street portion clipped to a fixed grid cell. It does not use home locations.
    """
    required = {cell_col, length_col, swi_col, dwi_col}
    missing = required.difference(clipped_segments.columns)
    if missing:
        raise ValueError(f"clipped_segments is missing columns: {sorted(missing)}")
    rows = []
    for cell_id, group in clipped_segments.groupby(cell_col, sort=False):
        rows.append(
            {
                cell_col: cell_id,
                "Neighborhood_SWI": length_weighted_mean(group[swi_col], group[length_col]),
                "Neighborhood_DWI": length_weighted_mean(group[dwi_col], group[length_col]),
                "street_length_m": float(group[length_col].clip(lower=0).sum()),
            }
        )
    return pd.DataFrame(rows)
