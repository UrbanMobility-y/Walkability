"""
Static and dynamic walkability indices.

DWI is treated as time varying vehicular copresence exposure rather than a full
measure of perceived traffic stress. Microscale separation features such as
continuous sidewalk width, guardrails, planted buffers, grade separated crossings
and curbside protection are not systematically observable in the public example
workflow or across the 82-city analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spatial_utils import haversine_distance_m, project_geometry_to_m


class StaticWalkabilityIndex:
    """Street level static walkability index using the eight adapted NetAScore indicators."""

    def __init__(self, weights: dict | None = None):
        self.weights = weights or {
            "road_hierarchy": 0.3,
            "gradient": 0.3,
            "building_density": 0.1,
            "amenity_availability": 0.3,
            "transit_accessibility": 0.3,
            "greenness_ratio": 0.3,
            "industrial_ratio": 0.3,
            "water_proximity": 0.4,
        }
        self.weight_sum = float(sum(self.weights.values()))
        self._sorted_raw_scores = None
        self.normalization_scope = None

    @staticmethod
    def equal_weights():
        return {k: 1.0 for k in [
            "road_hierarchy", "gradient", "building_density", "amenity_availability",
            "transit_accessibility", "greenness_ratio", "industrial_ratio", "water_proximity"
        ]}

    def score_road_hierarchy(self, road_type: str) -> float:
        mapping = {
            "motorway": 0.0, "trunk": 0.0, "primary": 0.0,
            "secondary": 0.2, "tertiary": 0.5,
            "residential": 0.8, "service": 0.85,
            "living_street": 0.9, "footway": 1.0, "pedestrian": 1.0,
        }
        if isinstance(road_type, (list, tuple)):
            road_type = road_type[0] if road_type else "residential"
        return mapping.get(str(road_type), 0.5)

    @staticmethod
    def score_gradient(slope_pct: float) -> float:
        slope = abs(float(slope_pct))
        if slope <= 3: return 1.0
        if slope <= 6: return 0.7
        if slope <= 12: return 0.5
        return 0.25

    @staticmethod
    def score_building_density(ratio: float) -> float:
        ratio = float(ratio)
        if ratio <= 0: return 1.0
        if ratio <= 0.2: return 0.8
        if ratio <= 0.4: return 0.6
        if ratio <= 0.6: return 0.4
        if ratio <= 0.8: return 0.2
        return 0.0

    @staticmethod
    def score_greenness(ratio: float) -> float:
        ratio = float(ratio)
        if ratio <= 0: return 0.0
        if ratio <= 0.05: return 0.7
        if ratio <= 0.50: return 0.8
        if ratio <= 0.75: return 0.9
        return 1.0

    @staticmethod
    def score_industrial(ratio: float) -> float:
        ratio = float(ratio)
        if ratio <= 0: return 1.0
        if ratio <= 0.05: return 0.3
        if ratio <= 0.50: return 0.2
        if ratio <= 0.75: return 0.1
        return 0.0

    def compute_raw_swi(self, segment_attr: dict) -> float:
        """Compute the adapted NetAScore weighted score before city ECDF scaling."""
        scores = {
            "road_hierarchy": self.score_road_hierarchy(segment_attr.get("highway", "residential")),
            "gradient": self.score_gradient(segment_attr.get("gradient", 0)),
            "building_density": self.score_building_density(segment_attr.get("bldg_density", 0)),
            "amenity_availability": 1.0 if segment_attr.get("has_amenity", False) else 0.0,
            "transit_accessibility": 1.0 if segment_attr.get("has_transit", False) else 0.0,
            "greenness_ratio": self.score_greenness(segment_attr.get("green_ratio", 0)),
            "industrial_ratio": self.score_industrial(segment_attr.get("industrial_ratio", segment_attr.get("ind_ratio", 0))),
            "water_proximity": 1.0 if segment_attr.get("near_water", False) else 0.0,
        }
        val = sum(scores[k] * self.weights[k] for k in scores) / self.weight_sum
        return float(np.clip(val, 0.0, 1.0))

    def compute_swi(self, segment_attr: dict) -> float:
        """Compatibility alias for the weighted score before ECDF conversion.

        The manuscript's *final* SWI additionally requires
        :meth:`fit_quantile_normalizer` and :meth:`get_swi` across all street
        segments in one city.
        """
        return self.compute_raw_swi(segment_attr)

    def fit_quantile_normalizer(self, raw_scores, scope: str = "within_city"):
        """Fit the city specific ECDF used to convert raw scores into final SWI."""
        values = np.asarray(raw_scores, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError("No finite raw SWI scores supplied.")
        self._sorted_raw_scores = np.sort(values)
        self.normalization_scope = scope
        return self

    def get_swi(self, raw_scores):
        """Return city specific empirical percentile positions in ``[0, 1]``."""
        if self._sorted_raw_scores is None:
            raise RuntimeError("Call fit_quantile_normalizer before get_swi.")
        values = np.asarray(raw_scores, dtype=float)
        return np.searchsorted(self._sorted_raw_scores, values, side="right") / len(self._sorted_raw_scores)


class DynamicWalkabilityIndex:
    """Vehicular exposure (VE), city specific quantile DWI, and conservative DCI."""

    def __init__(self, pcu_weights: dict | None = None):
        self.pcu_weights = pcu_weights or {"car": 1.0, "bus": 2.0, "truck": 2.5}
        self._sorted_exposure = None
        self.normalization_scope = None

    def compute_vehicular_exposure(self, car_density, bus_density, heavy_truck_density):
        """VE in passenger car equivalent units per kilometre."""
        return (np.asarray(car_density) * self.pcu_weights["car"] +
                np.asarray(bus_density) * self.pcu_weights["bus"] +
                np.asarray(heavy_truck_density) * self.pcu_weights["truck"])

    def fit_quantile_normalizer(self, vehicular_exposure_values, scope: str = "within_city"):
        """
        Fit ECDF normalizer.

        ``scope='within_city'`` reproduces the manuscript's city specific DWI.
        It supports within city percentile interpretation but does not preserve
        absolute between city traffic differences. Use raw VE or a global ECDF
        for absolute between city exposure comparisons.
        """
        values = np.asarray(vehicular_exposure_values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError("No finite vehicular exposure values supplied.")
        self._sorted_exposure = np.sort(values)
        self.normalization_scope = scope
        return self

    def get_dwi(self, vehicular_exposure):
        if self._sorted_exposure is None:
            raise RuntimeError("Call fit_quantile_normalizer before get_dwi.")
        values = np.asarray(vehicular_exposure, dtype=float)
        quantiles = np.searchsorted(self._sorted_exposure, values, side="right") / len(self._sorted_exposure)
        return 1.0 - quantiles

    @staticmethod
    def compute_dci(connected_dwis) -> float:
        """Conservative node level DCI: the least favourable connected DWI."""
        vals = np.asarray(connected_dwis, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return np.nan
        return float(np.min(vals))

    @classmethod
    def populate_graph_dci(cls, graph):
        """Recompute node DCI fields as minima over connected segment DWI fields."""
        for node in graph.nodes:
            incident = []
            if graph.is_directed():
                incident.extend(graph.in_edges(node, data=True))
                incident.extend(graph.out_edges(node, data=True))
            else:
                incident.extend(graph.edges(node, data=True))
            data_rows = [data for *_, data in incident]
            base = [d["DWI"] for d in data_rows if "DWI" in d]
            if base:
                graph.nodes[node]["DCI"] = cls.compute_dci(base)
            time_keys = {k[4:] for d in data_rows for k in d if k.startswith("DWI_")}
            schedule_keys = {
                k for d in data_rows if isinstance(d.get("dwi_schedule"), dict) for k in d["dwi_schedule"]
            }
            schedule = {}
            for time_key in time_keys | schedule_keys:
                vals = []
                for d in data_rows:
                    if f"DWI_{time_key}" in d:
                        vals.append(d[f"DWI_{time_key}"])
                    elif isinstance(d.get("dwi_schedule"), dict) and time_key in d["dwi_schedule"]:
                        vals.append(d["dwi_schedule"][time_key])
                    elif "DWI" in d:
                        vals.append(d["DWI"])
                if vals:
                    schedule[time_key] = cls.compute_dci(vals)
            if schedule:
                graph.nodes[node]["dci_schedule"] = schedule
        return graph


def estimate_intersection_dwell_time(connected_highways, crossing_required: bool) -> float:
    """Crossing dwell time based on the rules in Supplementary Note 5.2."""
    if not crossing_required:
        return 0.0
    road_types = {str(x) for x in connected_highways}
    if road_types & {"motorway", "trunk", "primary"}:
        return 61.0
    if road_types & {"secondary", "tertiary"}:
        return 37.5
    return 8.0


def pcu_weight_sensitivity(car_density, bus_density, truck_density,
                           schemes: dict | None = None) -> pd.DataFrame:
    """Compute VE under alternative PCU schemes."""
    schemes = schemes or {
        "conservative": {"car": 1.0, "bus": 1.5, "truck": 2.0},
        "baseline": {"car": 1.0, "bus": 2.0, "truck": 2.5},
        "aggressive": {"car": 1.0, "bus": 3.0, "truck": 4.5},
    }
    return pd.DataFrame({name: DynamicWalkabilityIndex(weights).compute_vehicular_exposure(car_density, bus_density, truck_density)
                         for name, weights in schemes.items()})


class PedestrianExperienceCalculator:
    """Route inferred PE_SWI, PE_DWI and PE_composite for paths in a NetworkX graph."""

    def __init__(self, graph, walk_speed_mps: float = 1.0):
        self.graph = graph
        self.walk_speed_mps = walk_speed_mps

    @staticmethod
    def _time_key(dt) -> str:
        dt = pd.to_datetime(dt)
        minute = (dt.minute // 5) * 5
        return f"{dt.hour:02d}:{minute:02d}"

    @staticmethod
    def _time_key_candidates(dt) -> tuple[str, str]:
        colon = PedestrianExperienceCalculator._time_key(dt)
        return colon, colon.replace(":", "")

    @staticmethod
    def _scheduled_value(data: dict, prefix: str, dt, default: float) -> float:
        colon, compact = PedestrianExperienceCalculator._time_key_candidates(dt)
        schedule = data.get(f"{prefix.lower()}_schedule")
        if isinstance(schedule, dict):
            for key in (colon, compact, "default"):
                if key in schedule:
                    return float(schedule[key])
        for key in (f"{prefix}_{colon}", f"{prefix}_{compact}"):
            if key in data:
                return float(data[key])
        return float(data.get(prefix, default))

    def _edge_data(self, u, v, edge_key=None):
        data = self.graph.get_edge_data(u, v)
        if data is None:
            raise ValueError(f"Missing edge {u}->{v}")
        if self.graph.is_multigraph():
            if edge_key is not None:
                if edge_key not in data:
                    raise ValueError(f"Missing edge {u}->{v} with key {edge_key!r}")
                return data[edge_key]
            if len(data) == 1:
                return next(iter(data.values()))
            raise ValueError(
                f"Path segment {u}->{v} has parallel edges; supply the selected edge key"
            )
        return data

    def _node_dci(self, node, when):
        d = self.graph.nodes[node]
        return self._scheduled_value(d, "DCI", when, np.nan)

    def compute_path_metrics(self, path_nodes: list, start_time, sigma: float = 0.5,
                             include_intersections: bool = True, edge_keys=None) -> dict:
        selected_keys = edge_keys if edge_keys is not None else getattr(path_nodes, "edge_keys", None)
        if selected_keys is not None and len(selected_keys) != max(0, len(path_nodes) - 1):
            raise ValueError("edge_keys must contain one key per path segment")
        current_time = pd.to_datetime(start_time)
        sw_sum = dw_sum = segment_time = dynamic_time = 0.0
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            edge_key = None if selected_keys is None else selected_keys[i]
            data = self._edge_data(u, v, edge_key=edge_key)
            length = float(data.get("length_m", data.get("length", 0.0)))
            travel_time = length / self.walk_speed_mps if self.walk_speed_mps > 0 else 0.0
            swi = float(data.get("SWI", 0.5))
            dwi = self._scheduled_value(data, "DWI", current_time, 0.5)
            sw_sum += swi * travel_time
            dw_sum += dwi * travel_time
            segment_time += travel_time
            dynamic_time += travel_time
            current_time += pd.to_timedelta(travel_time, unit="s")
            if include_intersections and i < len(path_nodes) - 2:
                j = path_nodes[i + 1]
                dwell = float(self.graph.nodes[j].get("dwell_time_s", 0.0))
                if dwell > 0:
                    dci = self._node_dci(j, current_time)
                    if np.isfinite(dci):
                        dw_sum += float(dci) * dwell
                        dynamic_time += dwell
                    current_time += pd.to_timedelta(dwell, unit="s")
        if segment_time == 0 or dynamic_time == 0:
            return {"PE_SWI": np.nan, "PE_DWI": np.nan, "PE_composite": np.nan}
        pe_swi = sw_sum / segment_time
        pe_dwi = dw_sum / dynamic_time
        return {
            "PE_SWI": pe_swi,
            "PE_DWI": pe_dwi,
            "PE_composite": (1.0 - sigma) * pe_swi + sigma * pe_dwi,
        }
