#!/usr/bin/env python3
"""Classify urban form from 0.005-degree population grid GeoJSON files.

All method settings are ordinary input parameters loaded from JSON.  The
program requires only NumPy; GIS, SciPy, and scikit-learn are not required.

Outputs
-------
urban_form_results.csv
    One row per city, including the computed metrics and final class.
urban_form_centers.geojson
    Primary and significant secondary population centers.
run_metadata.json
    Effective input parameters, formulas, record counts, and class counts.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


FORM_RADIAL = "Radial Monocentric"
FORM_CONCENTRIC = "Concentric Monocentric"
FORM_CLUSTERED = "Clustered Polycentric"
FORM_DISPERSED = "Dispersed Polycentric"
ALL_FORMS = (FORM_RADIAL, FORM_CONCENTRIC, FORM_CLUSTERED, FORM_DISPERSED)


@dataclass
class Config:
    # Inputs for the grid, smoothing and center detection
    grid_degrees: float = 0.005
    smoothing_sigma_km: float = 1.0
    smoothing_truncate_sigma: float = 3.5
    smoothing_boundary: str = "zero"  # or "mask_normalized"
    peak_radius_km: float = 1.0
    relative_center_threshold: float = 0.45
    min_center_separation_km: float = 2.0
    ring_width_km: float = 1.0
    sector_width_degrees: float = 15.0
    ann_threshold: float = 1.0
    spatial_concentration_threshold: float = 0.5

    # Inputs for the radial profile and concentration
    min_ring_population_share: float = 0.002
    radial_population_quantile: float = 0.95
    min_radial_rings: int = 4
    concentration_buffer_km: float = 6.0
    concentration_max_centers: int = 2

    # Inputs for classification scores
    polycentric_distance_scale_km: float = 25.0
    polycentric_prominence_power: float = 1.0
    radial_fit_weight: float = 0.25
    dispersed_center_count_weight: float = 1.0
    dispersed_ann_weight: float = 1.0
    dispersed_low_sc_weight: float = 1.0

    # Calibrated multiclass decision tree, supplied entirely through JSON.
    classification_tree: dict[str, Any] | None = None


def load_config(path: Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    with path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    valid = {field.name for field in fields(Config)}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {unknown}")
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


def validate_config(cfg: Config) -> None:
    positive = (
        "grid_degrees",
        "smoothing_sigma_km",
        "smoothing_truncate_sigma",
        "peak_radius_km",
        "min_center_separation_km",
        "ring_width_km",
        "sector_width_degrees",
        "concentration_buffer_km",
        "polycentric_distance_scale_km",
        "polycentric_prominence_power",
    )
    for key in positive:
        if float(getattr(cfg, key)) <= 0:
            raise ValueError(f"{key} must be positive")
    if cfg.smoothing_boundary not in {"mask_normalized", "zero"}:
        raise ValueError("smoothing_boundary must be 'mask_normalized' or 'zero'")
    if 360 % int(cfg.sector_width_degrees) != 0:
        raise ValueError("sector_width_degrees must divide 360")
    if int(cfg.concentration_max_centers) < 1:
        raise ValueError("concentration_max_centers must be at least 1")
    for key in (
        "relative_center_threshold",
        "spatial_concentration_threshold",
        "radial_population_quantile",
    ):
        value = float(getattr(cfg, key))
        if not 0 < value <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    finite_inputs = (
        "radial_fit_weight",
        "dispersed_center_count_weight",
        "dispersed_ann_weight",
        "dispersed_low_sc_weight",
    )
    for key in finite_inputs:
        if not math.isfinite(float(getattr(cfg, key))):
            raise ValueError(f"{key} must be finite")
    validate_classification_tree(cfg.classification_tree)


def city_name_from_path(path: Path) -> str:
    suffix = "_neighborhoods"
    return path.stem[: -len(suffix)] if path.stem.endswith(suffix) else path.stem


def polygon_center(coordinates: Any) -> tuple[float, float]:
    """Return a robust grid cell center for Polygon or MultiPolygon geometry."""
    points: list[tuple[float, float]] = []

    def collect(obj: Any) -> None:
        if (
            isinstance(obj, (list, tuple))
            and len(obj) >= 2
            and isinstance(obj[0], (int, float))
            and isinstance(obj[1], (int, float))
        ):
            points.append((float(obj[0]), float(obj[1])))
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                collect(item)

    collect(coordinates)
    if not points:
        raise ValueError("Geometry has no coordinates")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def load_city(path: Path, cfg: Config) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"{path.name}: expected a GeoJSON FeatureCollection")
    lons: list[float] = []
    lats: list[float] = []
    population: list[float] = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        if "pop_norm" not in props:
            raise ValueError(f"{path.name}: a feature is missing pop_norm")
        value = float(props["pop_norm"] or 0.0)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{path.name}: pop_norm must be finite and nonnegative")
        geometry = feature.get("geometry") or {}
        lon, lat = polygon_center(geometry.get("coordinates"))
        lons.append(lon)
        lats.append(lat)
        population.append(value)
    if not population:
        raise ValueError(f"{path.name}: no features")
    lons_a = np.asarray(lons, dtype=float)
    lats_a = np.asarray(lats, dtype=float)
    pop_a = np.asarray(population, dtype=float)
    pop_sum = float(pop_a.sum())
    if pop_sum <= 0:
        raise ValueError(f"{path.name}: total pop_norm is zero")
    pop_a /= pop_sum

    lat0 = float(np.mean(lats_a))
    km_per_lon_degree = 111.320 * math.cos(math.radians(lat0))
    km_per_lat_degree = 110.574
    cell_width_km = cfg.grid_degrees * km_per_lon_degree
    cell_height_km = cfg.grid_degrees * km_per_lat_degree
    cell_area_km2 = cell_width_km * cell_height_km

    # Integer grid locations are resistant to small serialization differences.
    lon_min = float(lons_a.min())
    lat_min = float(lats_a.min())
    cols = np.rint((lons_a - lon_min) / cfg.grid_degrees).astype(int)
    rows = np.rint((lats_a - lat_min) / cfg.grid_degrees).astype(int)
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    dense_pop = np.zeros(shape, dtype=float)
    dense_mask = np.zeros(shape, dtype=bool)
    dense_pop[rows, cols] = pop_a
    dense_mask[rows, cols] = True
    if int(dense_mask.sum()) != len(pop_a):
        raise ValueError(f"{path.name}: duplicate or invalid grid cell centers")

    return {
        "city": city_name_from_path(path),
        "source_file": str(path),
        "lons": lons_a,
        "lats": lats_a,
        "population": pop_a,
        "rows": rows,
        "cols": cols,
        "dense_pop": dense_pop,
        "dense_mask": dense_mask,
        "lat0": lat0,
        "cell_width_km": cell_width_km,
        "cell_height_km": cell_height_km,
        "cell_area_km2": cell_area_km2,
        "study_area_km2": cell_area_km2 * len(pop_a),
        "population_input_sum": pop_sum,
    }


def gaussian_kernel(sigma_cells: float, truncate: float) -> np.ndarray:
    radius = max(1, int(math.ceil(sigma_cells * truncate)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_cells) ** 2)
    return kernel / kernel.sum()


def convolve_axis_same(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """1-D convolution along one axis, preserving shape even for short axes."""
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(array, pad_width, mode="constant")
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), axis, padded)


def gaussian_smooth(city: dict[str, Any], cfg: Config) -> np.ndarray:
    sx = cfg.smoothing_sigma_km / city["cell_width_km"]
    sy = cfg.smoothing_sigma_km / city["cell_height_km"]
    kx = gaussian_kernel(sx, cfg.smoothing_truncate_sigma)
    ky = gaussian_kernel(sy, cfg.smoothing_truncate_sigma)
    numerator = convolve_axis_same(city["dense_pop"], kx, axis=1)
    numerator = convolve_axis_same(numerator, ky, axis=0)
    if cfg.smoothing_boundary == "zero":
        result = numerator
    else:
        denominator = convolve_axis_same(city["dense_mask"].astype(float), kx, axis=1)
        denominator = convolve_axis_same(denominator, ky, axis=0)
        result = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
    result[~city["dense_mask"]] = np.nan
    return result


def shifted_neighbor(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.full(array.shape, -np.inf, dtype=float)
    src_r0 = max(0, -dr)
    src_r1 = min(array.shape[0], array.shape[0] - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(array.shape[1], array.shape[1] - dc)
    dst_r0 = src_r0 + dr
    dst_r1 = src_r1 + dr
    dst_c0 = src_c0 + dc
    dst_c1 = src_c1 + dc
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0:dst_r1, dst_c0:dst_c1] = array[src_r0:src_r1, src_c0:src_c1]
    return out


def local_peak_candidates(city: dict[str, Any], smooth: np.ndarray, cfg: Config) -> list[dict[str, float]]:
    work = np.where(city["dense_mask"], smooth, -np.inf)
    peaks = city["dense_mask"].copy()
    max_dr = int(math.ceil(cfg.peak_radius_km / city["cell_height_km"]))
    max_dc = int(math.ceil(cfg.peak_radius_km / city["cell_width_km"]))
    for dr in range(-max_dr, max_dr + 1):
        for dc in range(-max_dc, max_dc + 1):
            if dr == 0 and dc == 0:
                continue
            distance = math.hypot(dr * city["cell_height_km"], dc * city["cell_width_km"])
            if distance <= cfg.peak_radius_km + 1e-12:
                peaks &= work >= shifted_neighbor(work, dr, dc) - 1e-15
    indices = np.argwhere(peaks)
    candidates: list[dict[str, float]] = []
    lon_min = float(city["lons"].min())
    lat_min = float(city["lats"].min())
    for row, col in indices:
        candidates.append(
            {
                "row": int(row),
                "col": int(col),
                "longitude": lon_min + col * cfg.grid_degrees,
                "latitude": lat_min + row * cfg.grid_degrees,
                "smoothed_population": float(work[row, col]),
            }
        )
    candidates.sort(key=lambda p: (-p["smoothed_population"], p["row"], p["col"]))
    if not candidates:
        flat_index = int(np.nanargmax(smooth))
        row, col = np.unravel_index(flat_index, smooth.shape)
        candidates = [
            {
                "row": int(row),
                "col": int(col),
                "longitude": lon_min + col * cfg.grid_degrees,
                "latitude": lat_min + row * cfg.grid_degrees,
                "smoothed_population": float(smooth[row, col]),
            }
        ]
    main = candidates[0]["smoothed_population"]
    for candidate in candidates:
        candidate["peak_ratio"] = candidate["smoothed_population"] / main if main > 0 else 0.0
    return candidates


def distance_km(p: dict[str, float], q: dict[str, float]) -> float:
    lat = (p["latitude"] + q["latitude"]) / 2.0
    dx = (p["longitude"] - q["longitude"]) * 111.320 * math.cos(math.radians(lat))
    dy = (p["latitude"] - q["latitude"]) * 110.574
    return math.hypot(dx, dy)


def suppress_nearby_peaks(candidates: list[dict[str, float]], cfg: Config) -> list[dict[str, float]]:
    kept: list[dict[str, float]] = []
    for candidate in candidates:
        if not kept or all(distance_km(candidate, other) > cfg.min_center_separation_km for other in kept):
            kept.append(dict(candidate))
    for rank, center in enumerate(kept, start=1):
        center["candidate_rank"] = rank
    return kept


def widest_connectivity_levels(
    smooth: np.ndarray, mask: np.ndarray, start: tuple[int, int]
) -> np.ndarray:
    """Maximum saddle level connecting every grid cell to the primary peak.

    This is a widest path traversal: the capacity of a path is its minimum
    smoothed value, and each cell receives the largest capacity over all paths
    from the primary peak.  The result measures the saddle height between a
    secondary peak and the primary core.
    """
    values = np.where(mask, smooth, -np.inf)
    best = np.full(values.shape, -np.inf, dtype=float)
    start_row, start_col = start
    best[start_row, start_col] = values[start_row, start_col]
    heap: list[tuple[float, int, int]] = [
        (-float(values[start_row, start_col]), int(start_row), int(start_col))
    ]
    while heap:
        negative_capacity, row, col = heapq.heappop(heap)
        capacity = -negative_capacity
        if capacity + 1e-18 < best[row, col]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                next_row, next_col = row + dr, col + dc
                if (
                    next_row < 0
                    or next_row >= values.shape[0]
                    or next_col < 0
                    or next_col >= values.shape[1]
                    or not mask[next_row, next_col]
                ):
                    continue
                new_capacity = min(capacity, float(values[next_row, next_col]))
                if new_capacity > best[next_row, next_col] + 1e-18:
                    best[next_row, next_col] = new_capacity
                    heapq.heappush(heap, (-new_capacity, next_row, next_col))
    return best


def annotate_peak_prominence(
    peaks: list[dict[str, float]], smooth: np.ndarray, mask: np.ndarray
) -> None:
    """Add saddle and topographic prominence ratios to retained peaks."""
    if not peaks:
        return
    main = peaks[0]
    main_value = float(main["smoothed_population"])
    connectivity = widest_connectivity_levels(smooth, mask, (int(main["row"]), int(main["col"])))
    for index, peak in enumerate(peaks):
        if index == 0:
            peak["saddle_ratio"] = 1.0
            peak["prominence_ratio"] = 0.0
            continue
        saddle_value = float(connectivity[int(peak["row"]), int(peak["col"])])
        if not math.isfinite(saddle_value):
            saddle_value = 0.0
        peak_value = float(peak["smoothed_population"])
        peak["saddle_ratio"] = saddle_value / main_value if main_value > 0 else 0.0
        peak["prominence_ratio"] = max(0.0, (peak_value - saddle_value) / main_value) if main_value > 0 else 0.0


def centers_at_alpha(peaks: list[dict[str, float]], alpha: float) -> list[dict[str, float]]:
    centers = [dict(p) for p in peaks if p["peak_ratio"] + 1e-12 >= alpha]
    if not centers and peaks:
        centers = [dict(peaks[0])]
    for rank, center in enumerate(centers, start=1):
        center["rank"] = rank
        center["center_type"] = "primary" if rank == 1 else "secondary"
    return centers


def cell_xy_km(city: dict[str, Any], lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    x = (city["lons"] - lon0) * 111.320 * math.cos(math.radians(lat0))
    y = (city["lats"] - lat0) * 110.574
    return x, y


def angular_cv_and_decay(
    city: dict[str, Any], smooth: np.ndarray, main_center: dict[str, float], cfg: Config
) -> dict[str, float]:
    lon0 = main_center["longitude"]
    lat0 = main_center["latitude"]
    x, y = cell_xy_km(city, lon0, lat0)
    distance = np.hypot(x, y)
    angle = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    values = smooth[city["rows"], city["cols"]]
    values = np.nan_to_num(values, nan=0.0)
    # Preserve a probability like scale after masked smoothing.
    if float(values.sum()) > 0:
        values = values / float(values.sum())

    ring = np.floor(distance / cfg.ring_width_km).astype(int)
    n_sectors = int(round(360.0 / cfg.sector_width_degrees))
    sector = np.floor(angle / cfg.sector_width_degrees).astype(int) % n_sectors

    # Only evaluate rings containing the central population mass.  This avoids
    # letting sparse cells near the administrative boundary dominate the mean CV.
    order = np.argsort(distance)
    cumulative = np.cumsum(values[order])
    cutoff_idx = int(np.searchsorted(cumulative, cfg.radial_population_quantile, side="left"))
    cutoff_idx = min(cutoff_idx, len(order) - 1)
    max_distance = max(float(distance[order[cutoff_idx]]), cfg.min_radial_rings * cfg.ring_width_km)
    max_ring = int(math.floor(max_distance / cfg.ring_width_km))

    ring_cvs: list[float] = []
    ring_weights: list[float] = []
    radial_d: list[float] = []
    radial_density: list[float] = []
    for ring_id in range(0, max_ring + 1):
        use = ring == ring_id
        ring_pop = float(values[use].sum())
        if ring_pop < cfg.min_ring_population_share:
            continue
        sector_pop = np.bincount(sector[use], weights=values[use], minlength=n_sectors).astype(float)
        sector_cells = np.bincount(sector[use], minlength=n_sectors).astype(float)
        observed = sector_cells > 0
        sector_density = sector_pop[observed] / sector_cells[observed]
        mean = float(sector_density.mean()) if len(sector_density) >= 5 else 0.0
        if mean > 0:
            ring_cvs.append(float(sector_density.std(ddof=0) / mean))
            ring_weights.append(ring_pop)
        inner = ring_id * cfg.ring_width_km
        outer = (ring_id + 1) * cfg.ring_width_km
        annulus_area = math.pi * (outer * outer - inner * inner)
        radial_d.append((inner + outer) / 2.0)
        radial_density.append(ring_pop / annulus_area)

    if ring_cvs:
        angular_cv = float(np.average(ring_cvs, weights=ring_weights))
    else:
        angular_cv = float("nan")

    d = np.asarray(radial_d, dtype=float)
    rho = np.asarray(radial_density, dtype=float)
    valid = (d > 0) & (rho > 0) & np.isfinite(rho)
    d = d[valid]
    rho = rho[valid]

    def fitted_r2(log_x: np.ndarray) -> float:
        if len(rho) < 3 or float(np.var(log_x)) <= 1e-15:
            return float("nan")
        slope, intercept = np.polyfit(log_x, np.log(rho), 1)
        prediction = np.exp(intercept + slope * log_x)
        denom = float(np.sum((rho - rho.mean()) ** 2))
        if denom <= 1e-20:
            return float("nan")
        return 1.0 - float(np.sum((rho - prediction) ** 2)) / denom

    r2_exp = fitted_r2(d)
    r2_power = fitted_r2(np.log(d))
    return {
        "angular_cv": angular_cv,
        "r2_exponential": r2_exp,
        "r2_power_law": r2_power,
        "r2_power_minus_exponential": r2_power - r2_exp,
        "radial_rings_used": int(len(d)),
    }


def average_nearest_neighbor(centers: Sequence[dict[str, float]], area_km2: float) -> float:
    if len(centers) < 2:
        return float("nan")
    nearest: list[float] = []
    for i, center in enumerate(centers):
        nearest.append(min(distance_km(center, other) for j, other in enumerate(centers) if i != j))
    observed = float(np.mean(nearest))
    expected = 0.5 * math.sqrt(max(area_km2, 1e-12) / len(centers))
    return observed / expected if expected > 0 else float("nan")


def convex_hull(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def point_segment_distance(
    x: np.ndarray, y: np.ndarray, a: tuple[float, float], b: tuple[float, float]
) -> np.ndarray:
    vx, vy = b[0] - a[0], b[1] - a[1]
    denom = vx * vx + vy * vy
    if denom <= 1e-15:
        return np.hypot(x - a[0], y - a[1])
    t = np.clip(((x - a[0]) * vx + (y - a[1]) * vy) / denom, 0.0, 1.0)
    return np.hypot(x - (a[0] + t * vx), y - (a[1] + t * vy))


def inside_polygon(x: np.ndarray, y: np.ndarray, polygon: Sequence[tuple[float, float]]) -> np.ndarray:
    if len(polygon) < 3:
        return np.zeros(x.shape, dtype=bool)
    inside = np.zeros(x.shape, dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-30) + xi
        )
        inside ^= crosses
        j = i
    return inside


def spatial_concentration(
    city: dict[str, Any], centers: Sequence[dict[str, float]], cfg: Config
) -> float:
    """Population share in a buffered hull around the dominant center cluster.

    A convex hull based on two points is a line with zero area, so the configurable
    buffer creates a measurable capsule shape around the strongest
    centers. An input with one center produces a buffered circle.
    """
    if not centers:
        return float("nan")
    cluster = list(centers[: int(cfg.concentration_max_centers)])
    main = cluster[0]
    x, y = cell_xy_km(city, main["longitude"], main["latitude"])
    center_xy: list[tuple[float, float]] = []
    for center in cluster:
        cx = (center["longitude"] - main["longitude"]) * 111.320 * math.cos(math.radians(main["latitude"]))
        cy = (center["latitude"] - main["latitude"]) * 110.574
        center_xy.append((cx, cy))
    hull = convex_hull(center_xy)
    if len(hull) == 1:
        distance_to_hull = np.hypot(x - hull[0][0], y - hull[0][1])
        selected = distance_to_hull <= cfg.concentration_buffer_km
    else:
        selected = inside_polygon(x, y, hull)
        min_distance = np.full(x.shape, np.inf, dtype=float)
        edges = list(zip(hull, hull[1:] + hull[:1])) if len(hull) >= 3 else [(hull[0], hull[1])]
        for a, b in edges:
            min_distance = np.minimum(min_distance, point_segment_distance(x, y, a, b))
        selected |= min_distance <= cfg.concentration_buffer_km
    return float(city["population"][selected].sum())


def analyze_city(path: Path, cfg: Config) -> dict[str, Any]:
    city = load_city(path, cfg)
    smooth = gaussian_smooth(city, cfg)
    local_peaks = local_peak_candidates(city, smooth, cfg)
    separated_peaks = suppress_nearby_peaks(local_peaks, cfg)
    annotate_peak_prominence(separated_peaks, smooth, city["dense_mask"])
    main = separated_peaks[0]
    mono_metrics = angular_cv_and_decay(city, smooth, main, cfg)
    max_secondary_ratio = separated_peaks[1]["peak_ratio"] if len(separated_peaks) > 1 else 0.0
    return {
        "city_data": city,
        "smooth": smooth,
        "peaks": separated_peaks,
        "city": city["city"],
        "source_file": city["source_file"],
        "n_grid_cells": int(len(city["population"])),
        "study_area_km2": float(city["study_area_km2"]),
        "population_input_sum": float(city["population_input_sum"]),
        "local_peak_count": int(len(local_peaks)),
        "separated_peak_count": int(len(separated_peaks)),
        "max_secondary_peak_ratio": float(max_secondary_ratio),
        **mono_metrics,
    }


def safe_number(value: float, fallback: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else fallback


TREE_RAW_FEATURES = {
    "polycentricity_score",
    "max_peak_prominence_ratio",
    "max_secondary_peak_ratio",
    "angular_cv",
    "r2_exponential",
    "r2_power_law",
    "r2_power_minus_exponential",
    "study_area_km2",
    "local_peak_count",
    "separated_peak_count",
    "radial_rings_used",
}


def validate_classification_tree(model: dict[str, Any] | None) -> None:
    """Validate a JSON decision tree without accepting executable expressions."""
    if not isinstance(model, dict) or not isinstance(model.get("tree"), dict):
        raise ValueError("classification_tree must contain a tree object")

    def visit(node: dict[str, Any], depth: int) -> None:
        if depth > 32:
            raise ValueError("classification_tree exceeds the maximum supported depth (32)")
        if "label" in node:
            if node["label"] not in ALL_FORMS:
                raise ValueError(f"Unknown tree leaf label: {node['label']!r}")
            return
        feature = node.get("feature")
        raw_feature = feature[6:] if isinstance(feature, str) and feature.startswith("log1p_") else feature
        if raw_feature not in TREE_RAW_FEATURES:
            raise ValueError(f"Unsupported classification tree feature: {feature!r}")
        threshold = node.get("threshold")
        if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
            raise ValueError("Every split threshold must be finite")
        if not isinstance(node.get("left"), dict) or not isinstance(node.get("right"), dict):
            raise ValueError("Every split must contain left and right child objects")
        visit(node["left"], depth + 1)
        visit(node["right"], depth + 1)

    visit(model["tree"], 0)


def tree_feature_value(feature: str, values: dict[str, float]) -> float:
    if feature.startswith("log1p_"):
        raw_value = safe_number(values[feature[6:]])
        return math.log1p(max(raw_value, 0.0))
    return safe_number(values[feature])


def classify_fixed_tree(model: dict[str, Any], values: dict[str, float]) -> str:
    """Evaluate the frozen numeric tree; no city name or cohort count is used."""
    node = model["tree"]
    while "label" not in node:
        value = tree_feature_value(node["feature"], values)
        node = node["left"] if value <= float(node["threshold"]) else node["right"]
    return str(node["label"])


def classify_all(analyses: list[dict[str, Any]], cfg: Config) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_centers: list[dict[str, Any]] = []
    for analysis in analyses:
        city_name = analysis["city"]
        main = analysis["peaks"][0]
        branch_score = max(
            (
                peak["peak_ratio"]
                * min(distance_km(main, peak) / cfg.polycentric_distance_scale_km, 1.0)
                * peak["prominence_ratio"] ** cfg.polycentric_prominence_power
                for peak in analysis["peaks"][1:]
            ),
            default=0.0,
        )
        maximum_prominence = max(
            (peak["prominence_ratio"] for peak in analysis["peaks"][1:]),
            default=0.0,
        )
        tree_values = {
            key: safe_number(analysis.get(key, 0.0))
            for key in TREE_RAW_FEATURES
            if key not in {"polycentricity_score", "max_peak_prominence_ratio"}
        }
        tree_values["polycentricity_score"] = branch_score
        tree_values["max_peak_prominence_ratio"] = maximum_prominence
        final_form = classify_fixed_tree(cfg.classification_tree, tree_values)  # type: ignore[arg-type]
        final_branch = "monocentric" if final_form in {FORM_RADIAL, FORM_CONCENTRIC} else "polycentric"
        threshold_fallback = False
        if final_branch == "monocentric":
            centers = centers_at_alpha(analysis["peaks"][:1], cfg.relative_center_threshold)
        else:
            centers = centers_at_alpha(analysis["peaks"], cfg.relative_center_threshold)
            if len(centers) == 1:
                threshold_fallback = True
                if len(analysis["peaks"]) >= 2:
                    centers = centers_at_alpha(analysis["peaks"][:2], 0.0)
        n = len(centers)
        row = {key: value for key, value in analysis.items() if key not in {"city_data", "smooth", "peaks"}}
        row["relative_center_threshold"] = cfg.relative_center_threshold
        row["polycentricity_score"] = branch_score
        row["max_peak_prominence_ratio"] = maximum_prominence
        row["n_centers"] = n
        row["n_secondary_centers"] = max(0, n - 1)
        row["primary_longitude"] = centers[0]["longitude"]
        row["primary_latitude"] = centers[0]["latitude"]
        row["branch"] = final_branch
        row["center_threshold_fallback_used"] = threshold_fallback
        row["ann_ratio"] = average_nearest_neighbor(centers, analysis["city_data"]["study_area_km2"])
        row["spatial_concentration"] = spatial_concentration(analysis["city_data"], centers, cfg)
        row["radial_score"] = float("nan")
        row["dispersed_score"] = float("nan")
        if final_branch == "monocentric":
            row["radial_score"] = safe_number(row["angular_cv"]) + cfg.radial_fit_weight * safe_number(
                row["r2_power_minus_exponential"]
            )
        else:
            center_component = max(n - 3, 0)
            ann_component = safe_number(row["ann_ratio"]) - cfg.ann_threshold
            low_sc_component = cfg.spatial_concentration_threshold - safe_number(row["spatial_concentration"])
            row["dispersed_score"] = (
                cfg.dispersed_center_count_weight * center_component
                + cfg.dispersed_ann_weight * ann_component
                + cfg.dispersed_low_sc_weight * low_sc_component
            )
        row["urban_form"] = final_form
        row["classification_source"] = "fixed_parameter_depth_7_tree"
        rows.append(row)
        for center in centers:
            all_centers.append(
                {
                    "city": city_name,
                    "urban_form": row["urban_form"],
                    "branch": final_branch,
                    **center,
                }
            )

    counts = Counter(row["urban_form"] for row in rows)
    report = {
        "city_count": len(rows),
        "class_counts": {form: counts.get(form, 0) for form in ALL_FORMS},
        "formulas": {
            "polycentricity_score": "max_secondary(peak_ratio * min(distance_to_primary / polycentric_distance_scale_km, 1) * prominence_ratio^polycentric_prominence_power)",
            "radial_score": "angular_cv + radial_fit_weight * (r2_power_law - r2_exponential)",
            "dispersed_score": "w_center*max(n_centers-3,0) + w_ann*(ann_ratio-ann_threshold) + w_sc*(spatial_concentration_threshold-SC)",
        },
        "classification_model": cfg.classification_tree.get("model_type"),  # type: ignore[union-attr]
        "classification_tree_maximum_depth": cfg.classification_tree.get("maximum_depth"),  # type: ignore[union-attr]
        "all_centers": all_centers,
    }
    return rows, report


CSV_COLUMNS = [
    "city",
    "urban_form",
    "branch",
    "n_centers",
    "n_secondary_centers",
    "relative_center_threshold",
    "polycentricity_score",
    "max_peak_prominence_ratio",
    "max_secondary_peak_ratio",
    "angular_cv",
    "r2_exponential",
    "r2_power_law",
    "r2_power_minus_exponential",
    "radial_score",
    "ann_ratio",
    "spatial_concentration",
    "dispersed_score",
    "classification_source",
    "center_threshold_fallback_used",
    "primary_longitude",
    "primary_latitude",
    "study_area_km2",
    "n_grid_cells",
    "population_input_sum",
    "local_peak_count",
    "separated_peak_count",
    "radial_rings_used",
    "source_file",
]


def clean_json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_outputs(rows: list[dict[str, Any]], report: dict[str, Any], output_dir: Path, cfg: Config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "urban_form_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["city"]):
            writer.writerow({key: clean_json_value(value) for key, value in row.items()})

    center_features = []
    for center in report.pop("all_centers"):
        properties = {key: clean_json_value(value) for key, value in center.items() if key not in {"longitude", "latitude", "row", "col"}}
        center_features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Point",
                    "coordinates": [center["longitude"], center["latitude"]],
                },
            }
        )
    centers_geojson = {
        "type": "FeatureCollection",
        "name": "urban_form_centers",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": center_features,
    }
    with (output_dir / "urban_form_centers.geojson").open("w", encoding="utf-8") as stream:
        json.dump(centers_geojson, stream, ensure_ascii=False, indent=2, allow_nan=False)

    report["configuration"] = asdict(cfg)
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "source data" / "neighborhood-level walkability",
        help="Directory containing *_neighborhoods.json (default: sibling source data directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("urban_form_results"),
        help="Output directory (default: ./urban_form_results)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "urban_form_config.json",
        help="JSON configuration (default: configs/urban_form_config.json)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    validate_config(cfg)
    paths = sorted(args.input_dir.glob("*_neighborhoods.json"))
    if not paths:
        raise FileNotFoundError(f"No *_neighborhoods.json files found in {args.input_dir}")
    analyses: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index:02d}/{len(paths):02d}] {city_name_from_path(path)}", file=sys.stderr)
        analyses.append(analyze_city(path, cfg))
    rows, report = classify_all(analyses, cfg)
    write_outputs(rows, report, args.output_dir, cfg)
    counts = Counter(row["urban_form"] for row in rows)
    print(json.dumps({form: counts.get(form, 0) for form in ALL_FORMS}, ensure_ascii=False))
    print(f"Results written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
