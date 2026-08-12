"""Agent based route choice modelling and residential policy scenarios.

The implementation follows the equations and scenario boundaries in the
manuscript. In particular, SWI/DWI
perturbations are applied separately for each resident and only to segments in
that resident's home centered 15-minute catchment. Raw vehicle fields, trip
demand, land use and system level feedback remain fixed.
"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from neighborhood_walkability import catchment_edge_keys
from walkability_indices import DynamicWalkabilityIndex, PedestrianExperienceCalculator


def draw_stratified_calibration_sample(
    trips: pd.DataFrame,
    sample_size: int = 500_000,
    distance_col: str = "route_length_m",
    departure_col: str = "departure_time",
    spatial_origin_col: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Draw the calibration sample stratified by distance and time in Supplementary Note 6.1.2.

    Distance targets are 30% (0--500 m), 40% (500--1,000 m), 25%
    (1--2 km) and 5% (>2 km). Peak periods (07:00--09:00 and
    17:00--19:00) receive 40% total target weight. If a stratum for spatial
    origin is supplied, its observed proportions are retained.
    """
    required = {distance_col, departure_col}
    missing = required.difference(trips.columns)
    if missing:
        raise ValueError(f"calibration trips are missing columns: {sorted(missing)}")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(trips) <= sample_size:
        return trips.copy()

    frame = trips.copy()
    frame["_distance_band"] = pd.cut(
        pd.to_numeric(frame[distance_col], errors="coerce"),
        bins=[0, 500, 1000, 2000, np.inf],
        labels=["0_500", "500_1000", "1000_2000", "over_2000"],
        include_lowest=True,
        right=False,
    )
    hours = pd.to_datetime(frame[departure_col], errors="coerce").dt.hour
    frame["_time_band"] = np.where(hours.isin([7, 8, 17, 18]), "peak", "off_peak")
    frame = frame.dropna(subset=["_distance_band"])
    distance_targets = {"0_500": 0.30, "500_1000": 0.40, "1000_2000": 0.25, "over_2000": 0.05}
    time_targets = {"peak": 0.40, "off_peak": 0.60}
    distance_observed = frame["_distance_band"].astype(str).value_counts(normalize=True)
    time_observed = frame["_time_band"].value_counts(normalize=True)
    frame["_sample_weight"] = [
        distance_targets[str(distance)] / distance_observed[str(distance)]
        * time_targets[time] / time_observed[time]
        for distance, time in zip(frame["_distance_band"], frame["_time_band"])
    ]

    if spatial_origin_col is None:
        sampled = frame.sample(
            n=min(sample_size, len(frame)), weights="_sample_weight", replace=False, random_state=random_state
        )
    else:
        if spatial_origin_col not in frame.columns:
            raise ValueError(f"missing spatial origin stratum column: {spatial_origin_col}")
        pieces = []
        rng = np.random.default_rng(random_state)
        proportions = frame[spatial_origin_col].value_counts(normalize=True, dropna=False)
        allocations = (proportions * sample_size).round().astype(int)
        # Correct rounding so allocations sum to the requested size.
        allocations.iloc[0] += sample_size - int(allocations.sum())
        for stratum, requested in allocations.items():
            if pd.isna(stratum):
                group = frame[frame[spatial_origin_col].isna()]
            else:
                group = frame[frame[spatial_origin_col] == stratum]
            pieces.append(
                group.sample(
                    n=min(int(requested), len(group)),
                    weights="_sample_weight",
                    replace=False,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
            )
        sampled = pd.concat(pieces)
        if len(sampled) < sample_size:
            remaining = frame.drop(index=sampled.index)
            extra = remaining.sample(
                n=min(sample_size - len(sampled), len(remaining)),
                weights="_sample_weight",
                replace=False,
                random_state=random_state + 1,
            )
            sampled = pd.concat([sampled, extra])
    return sampled.drop(columns=["_distance_band", "_time_band", "_sample_weight"]).sample(
        frac=1, random_state=random_state
    )


class SegmentCostFunction:
    """Exponentially penalized traversal cost ``L * exp(beta * (1 - WI))``."""

    def __init__(self, beta: float = 1.8):
        self.beta = float(beta)

    def compute(self, length_m: float, walkability_index: float) -> float:
        wi = float(np.clip(walkability_index, 0.0, 1.0))
        return float(length_m * np.exp(self.beta * (1.0 - wi)))


class RoutePath(list):
    """Node path with the selected graph edge key for every segment.

    The class remains list compatible for code that consumes node paths. The
    ``edge_keys`` and ``keyed_edges`` attributes retain parallel edge identity
    when the source network is a MultiGraph or MultiDiGraph.
    """

    def __init__(self, nodes, edge_keys):
        super().__init__(nodes)
        self.edge_keys = tuple(edge_keys)
        if len(self.edge_keys) != max(0, len(self) - 1):
            raise ValueError("edge_keys must contain one key per path segment")

    @property
    def keyed_edges(self) -> tuple[tuple, ...]:
        return tuple(
            (self[index], self[index + 1], key)
            for index, key in enumerate(self.edge_keys)
        )

    def __repr__(self) -> str:
        return f"RoutePath(nodes={list(self)!r}, edge_keys={self.edge_keys!r})"


class RouteOptimizer:
    """Minimum cost routing under a composite within city SWI/DWI surface."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph

    @staticmethod
    def _time_candidates(time_period) -> tuple[str, ...]:
        if time_period is None or (isinstance(time_period, float) and np.isnan(time_period)):
            return ()
        text = str(time_period)
        if ":" in text:
            compact = text.replace(":", "")
            return text, compact
        if len(text) == 4 and text.isdigit():
            return text, f"{text[:2]}:{text[2:]}"
        return (text,)

    @classmethod
    def _dwi_from_edge(cls, data: dict, time_period=None) -> float:
        candidates = cls._time_candidates(time_period)
        schedule = data.get("dwi_schedule")
        if isinstance(schedule, dict):
            for key in (*candidates, "default"):
                if key in schedule:
                    return float(schedule[key])
        for key in candidates:
            if f"DWI_{key}" in data:
                return float(data[f"DWI_{key}"])
        return float(data.get("DWI", 0.5))

    def _weighted_graph(self, sigma: float, beta: float, time_period=None) -> nx.DiGraph:
        if not 0 <= sigma <= 1:
            raise ValueError("sigma must lie in [0, 1]")
        graph = nx.DiGraph()
        graph.add_nodes_from(self.graph.nodes)
        cost = SegmentCostFunction(beta)
        if self.graph.is_multigraph():
            edges = self.graph.edges(keys=True, data=True)
        else:
            edges = ((u, v, None, data) for u, v, data in self.graph.edges(data=True))
        for u, v, edge_key, data in edges:
            length = float(data.get("length_m", data.get("length", 1.0)))
            swi = float(data.get("SWI", 0.5))
            dwi = self._dwi_from_edge(data, time_period)
            wi = (1.0 - sigma) * swi + sigma * dwi
            weight = cost.compute(length, wi)
            if not graph.has_edge(u, v) or weight < graph[u][v]["weight"]:
                graph.add_edge(u, v, weight=weight, edge_key=edge_key)
            if not self.graph.is_directed():
                if not graph.has_edge(v, u) or weight < graph[v][u]["weight"]:
                    graph.add_edge(v, u, weight=weight, edge_key=edge_key)
        return graph

    def route(self, origin, destination, sigma: float, beta: float, time_period=None):
        graph = self._weighted_graph(sigma, beta, time_period=time_period)
        try:
            nodes = nx.dijkstra_path(graph, origin, destination, weight="weight")
            edge_keys = [graph[u][v].get("edge_key") for u, v in zip(nodes[:-1], nodes[1:])]
            return RoutePath(nodes, edge_keys)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


class BehavioralModelCalibration:
    """Calibrate sigma and beta against route inferred PE benchmarks."""

    def __init__(self, observed_trips: pd.DataFrame, street_network: nx.Graph):
        self.observed_trips = observed_trips.copy()
        self.graph = street_network

    def _mse_for_params(
        self,
        sigma: float,
        beta: float,
        trips: pd.DataFrame,
        lambda_swi: float = 0.5,
    ) -> float:
        """Combined loss; lambda=0.5 equals ``e_SWI + e_DWI`` up to no scaling."""
        if not 0 <= lambda_swi <= 1:
            raise ValueError("lambda_swi must lie in [0, 1]")
        optimizer = RouteOptimizer(self.graph)
        calculator = PedestrianExperienceCalculator(self.graph)
        errors = []
        for _, trip in trips.iterrows():
            path = optimizer.route(
                trip["origin_node"], trip["dest_node"], sigma, beta, trip.get("time_period")
            )
            if path is None:
                continue
            predicted = calculator.compute_path_metrics(
                path, trip.get("departure_time", "2024-09-01 08:00"), sigma=sigma
            )
            error_swi = (predicted["PE_SWI"] - trip.get("observed_PE_SWI", np.nan)) ** 2
            error_dwi = (predicted["PE_DWI"] - trip.get("observed_PE_DWI", np.nan)) ** 2
            if np.isfinite(error_swi) and np.isfinite(error_dwi):
                # Multiplication by two makes the 50/50 convention exactly the
                # manuscript's sum of the two squared errors.
                errors.append(2.0 * (lambda_swi * error_swi + (1.0 - lambda_swi) * error_dwi))
        return float(np.mean(errors)) if errors else np.inf

    def grid_search_parameters(
        self,
        sigma_values=np.arange(0, 1.01, 0.05),
        beta_values=np.arange(0, 5.01, 0.1),
        sample_size: int | None = None,
        lambda_swi: float = 0.5,
        random_state: int = 42,
        stratified_sample: bool = False,
        distance_col: str = "route_length_m",
        departure_col: str = "departure_time",
        spatial_origin_col: str | None = None,
    ):
        trips = self.observed_trips
        if sample_size is not None and len(trips) > sample_size:
            if stratified_sample:
                trips = draw_stratified_calibration_sample(
                    trips,
                    sample_size=sample_size,
                    distance_col=distance_col,
                    departure_col=departure_col,
                    spatial_origin_col=spatial_origin_col,
                    random_state=random_state,
                )
            else:
                trips = trips.sample(sample_size, random_state=random_state)
        mse = np.zeros((len(sigma_values), len(beta_values)))
        for i, sigma in enumerate(sigma_values):
            for j, beta in enumerate(beta_values):
                mse[i, j] = self._mse_for_params(
                    float(sigma), float(beta), trips, lambda_swi=lambda_swi
                )
        index = np.unravel_index(np.nanargmin(mse), mse.shape)
        params = {
            "sigma": float(sigma_values[index[0]]),
            "beta": float(beta_values[index[1]]),
            "min_mse": float(mse[index]),
        }
        return params, mse

    @staticmethod
    def parameter_envelope(
        mse_surface: np.ndarray,
        sigma_values,
        beta_values,
        tolerance: float = 1.05,
    ) -> pd.DataFrame:
        """Return parameter pairs satisfying MSE <= tolerance * minimum MSE."""
        minimum = float(np.nanmin(mse_surface))
        rows = []
        for i, sigma in enumerate(sigma_values):
            for j, beta in enumerate(beta_values):
                if mse_surface[i, j] <= tolerance * minimum:
                    rows.append(
                        {"sigma": float(sigma), "beta": float(beta), "mse": float(mse_surface[i, j])}
                    )
        return pd.DataFrame(rows)

    def validate(self, sigma: float, beta: float, trips: pd.DataFrame | None = None) -> dict:
        trips = self.observed_trips if trips is None else trips
        optimizer = RouteOptimizer(self.graph)
        calculator = PedestrianExperienceCalculator(self.graph)
        predicted_swi, predicted_dwi, observed_swi, observed_dwi = [], [], [], []
        for _, trip in trips.iterrows():
            path = optimizer.route(
                trip["origin_node"], trip["dest_node"], sigma, beta, trip.get("time_period")
            )
            if path is None:
                continue
            predicted = calculator.compute_path_metrics(
                path, trip.get("departure_time", "2024-09-01 08:00"), sigma=sigma
            )
            predicted_swi.append(predicted["PE_SWI"])
            predicted_dwi.append(predicted["PE_DWI"])
            observed_swi.append(trip["observed_PE_SWI"])
            observed_dwi.append(trip["observed_PE_DWI"])
        return {
            "pearson_r_SWI": float(pearsonr(observed_swi, predicted_swi)[0]) if len(observed_swi) > 1 else np.nan,
            "spearman_rho_SWI": float(spearmanr(observed_swi, predicted_swi)[0]) if len(observed_swi) > 1 else np.nan,
            "pearson_r_DWI": float(pearsonr(observed_dwi, predicted_dwi)[0]) if len(observed_dwi) > 1 else np.nan,
            "spearman_rho_DWI": float(spearmanr(observed_dwi, predicted_dwi)[0]) if len(observed_dwi) > 1 else np.nan,
            "n_trips": int(len(observed_swi)),
        }


class PolicyScenarioSimulator:
    """Residentially anchored, partial equilibrium counterfactual scenarios."""

    METRICS = ("PE_SWI", "PE_DWI", "PE_composite")

    def __init__(
        self,
        street_network: nx.Graph,
        observed_trips: pd.DataFrame,
        sigma: float,
        beta: float,
        default_catchment_radius_m: float | None = None,
        stratum_col: str = "neighborhood_type",
    ):
        self.base_graph = street_network
        self.trips = observed_trips.copy()
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.default_catchment_radius_m = default_catchment_radius_m
        self.stratum_col = stratum_col

    def _simulate_trips(
        self,
        trips: pd.DataFrame,
        graph_factory: Callable[[pd.Series], nx.Graph] | None = None,
    ) -> pd.DataFrame:
        rows = []
        for trip_index, trip in trips.iterrows():
            graph = self.base_graph if graph_factory is None else graph_factory(trip)
            optimizer = RouteOptimizer(graph)
            calculator = PedestrianExperienceCalculator(graph)
            destination = trip.get("simulated_dest", trip["dest_node"])
            path = optimizer.route(
                trip["origin_node"], destination, self.sigma, self.beta, trip.get("time_period")
            )
            if path is None:
                continue
            metrics = calculator.compute_path_metrics(
                path, trip.get("departure_time", "2024-09-01 08:00"), sigma=self.sigma
            )
            metrics["trip_index"] = trip_index
            if self.stratum_col in trips.columns:
                metrics["stratum"] = str(trip[self.stratum_col])
            rows.append(metrics)
        return pd.DataFrame(rows)

    def _aggregate(self, simulated: pd.DataFrame) -> pd.DataFrame:
        if simulated.empty:
            return pd.DataFrame(columns=["stratum", *self.METRICS, "n_trips"])
        rows = []
        all_row = simulated[list(self.METRICS)].mean().to_dict()
        all_row.update({"stratum": "all", "n_trips": int(len(simulated))})
        rows.append(all_row)
        if "stratum" in simulated.columns:
            for stratum, group in simulated.groupby("stratum", sort=True):
                row = group[list(self.METRICS)].mean().to_dict()
                row.update({"stratum": str(stratum), "n_trips": int(len(group))})
                rows.append(row)
        return pd.DataFrame(rows)

    def _baseline_table(self) -> pd.DataFrame:
        return self._aggregate(self._simulate_trips(self.trips))

    def baseline(self) -> dict:
        """Return baseline means across all trips (legacy convenience API)."""
        table = self._baseline_table()
        return table.loc[table["stratum"] == "all"].iloc[0].to_dict() if not table.empty else {}

    @staticmethod
    def _safe_percent_change(value: float, baseline: float) -> float:
        if not np.isfinite(value) or not np.isfinite(baseline) or baseline == 0:
            return np.nan
        return float(100.0 * (value - baseline) / baseline)

    def _add_changes(self, result: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
        baseline_by_stratum = baseline.set_index("stratum")
        result = result.copy()
        for metric in self.METRICS:
            result[f"delta_{metric}_pct"] = [
                self._safe_percent_change(row[metric], baseline_by_stratum.at[row["stratum"], metric])
                for _, row in result.iterrows()
            ]
        return result

    def _treated_graph(self, trip: pd.Series, alpha: float, dimension: str) -> nx.Graph:
        graph = self.base_graph.copy()
        treated = catchment_edge_keys(graph, trip, self.default_catchment_radius_m)
        if not treated:
            raise ValueError(f"no graph edges intersect the catchment for trip {trip.name!r}")
        for u, v, key in treated:
            data = graph[u][v] if key is None else graph[u][v][key]
            if dimension == "SWI":
                value = float(data.get("SWI", 0.5))
                data["SWI"] = value + (1.0 - value) * float(alpha)
            elif dimension == "DWI":
                for field in list(data):
                    if field == "DWI" or field.startswith("DWI_"):
                        value = float(data[field])
                        data[field] = value + (1.0 - value) * float(alpha)
                if isinstance(data.get("dwi_schedule"), dict):
                    data["dwi_schedule"] = {
                        time_key: float(value) + (1.0 - float(value)) * float(alpha)
                        for time_key, value in data["dwi_schedule"].items()
                    }
            else:
                raise ValueError(f"unknown perturbation dimension: {dimension}")
        if dimension == "DWI":
            DynamicWalkabilityIndex.populate_graph_dci(graph)
        return graph

    def scenario_amenity_localization(
        self,
        alpha_values=np.arange(0, 1.01, 0.1),
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Redirect a nested proportion of home based extra neighborhood trips."""
        baseline = self._baseline_table()
        if "is_extra_neighborhood" not in self.trips.columns:
            raise ValueError("amenity localization requires is_extra_neighborhood")
        eligible_mask = self.trips["is_extra_neighborhood"].astype(bool)
        if "is_home_based" in self.trips.columns:
            eligible_mask &= self.trips["is_home_based"].astype(bool)
        eligible = self.trips.index[eligible_mask].to_numpy()
        rng = np.random.default_rng(random_state)
        order = rng.permutation(eligible)
        local_destination = {}
        for index in eligible:
            candidates = self.trips.at[index, "local_candidates"]
            if isinstance(candidates, (list, tuple, np.ndarray)) and len(candidates):
                local_destination[index] = rng.choice(candidates)
        order = np.asarray([index for index in order if index in local_destination])

        outputs = []
        for alpha in alpha_values:
            if not 0 <= float(alpha) <= 1:
                raise ValueError("scenario intensities must lie in [0, 1]")
            trips = self.trips.copy()
            trips["simulated_dest"] = trips["dest_node"]
            count = int(np.floor(float(alpha) * len(order) + 0.5))
            selected = order[:count]
            for index in selected:
                trips.at[index, "simulated_dest"] = local_destination[index]
            result = self._add_changes(self._aggregate(self._simulate_trips(trips)), baseline)
            result["alpha"] = float(alpha)
            result["scenario"] = "amenity_localization"
            result["redirected_trips"] = int(len(selected))
            outputs.append(result)
        return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()

    def scenario_built_environment_improvement(
        self, alpha_values=np.arange(0, 1.01, 0.1)
    ) -> pd.DataFrame:
        """Shift catchment SWI percentiles toward 1 while holding DWI fixed."""
        return self._index_perturbation("SWI", "built_environment", alpha_values)

    def scenario_traffic_calming(
        self, alpha_values=np.arange(0, 1.01, 0.1)
    ) -> pd.DataFrame:
        """Shift catchment DWI percentiles toward 1; raw traffic remains fixed."""
        return self._index_perturbation("DWI", "traffic_calming_index", alpha_values)

    def _index_perturbation(self, dimension: str, scenario: str, alpha_values) -> pd.DataFrame:
        baseline = self._baseline_table()
        outputs = []
        for alpha in alpha_values:
            if not 0 <= float(alpha) <= 1:
                raise ValueError("scenario intensities must lie in [0, 1]")
            result = self._aggregate(
                self._simulate_trips(
                    self.trips,
                    graph_factory=lambda trip, a=float(alpha): self._treated_graph(trip, a, dimension),
                )
            )
            result = self._add_changes(result, baseline)
            result["alpha"] = float(alpha)
            result["scenario"] = scenario
            outputs.append(result)
        return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def run_parameter_sensitivity(
    street_network,
    observed_trips,
    parameter_envelope: pd.DataFrame,
    alpha_values=(0, 0.5, 1.0),
    default_catchment_radius_m: float | None = None,
) -> pd.DataFrame:
    """Repeat all three scenarios over the <=5% MSE parameter envelope."""
    rows = []
    for _, parameters in parameter_envelope.iterrows():
        simulator = PolicyScenarioSimulator(
            street_network,
            observed_trips,
            parameters["sigma"],
            parameters["beta"],
            default_catchment_radius_m=default_catchment_radius_m,
        )
        for method in (
            simulator.scenario_amenity_localization,
            simulator.scenario_built_environment_improvement,
            simulator.scenario_traffic_calming,
        ):
            frame = method(alpha_values=alpha_values)
            frame["sigma"] = parameters["sigma"]
            frame["beta"] = parameters["beta"]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
