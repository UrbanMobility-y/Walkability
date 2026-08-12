"""Run a small complete reproducibility demonstration with synthetic data.

This script does not call AMap and does not require restricted raw data. It
executes the main public code components on bundled example data and writes
outputs to ``outputs/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mobility_data_processing import process_user_trajectory
from route_reconstruction import validate_route_geometry, route_quality_sensitivity
from transport_mode_inference import walking_threshold_sensitivity
from truck_data_processing import identify_truck_stops_adaptive
from walkability_indices import StaticWalkabilityIndex, DynamicWalkabilityIndex
from behavioral_modeling_policy import BehavioralModelCalibration, PolicyScenarioSimulator


def build_demo_graph():
    G = nx.DiGraph()
    coordinates = {"A": (0, 0), "B": (90, 0), "C": (210, 0), "D": (0, 180), "E": (350, 180)}
    for node, (x, y) in coordinates.items():
        G.add_node(node, x=x, y=y, dwell_time_s=0, DCI=0.7)
    edges = [
        ("A", "B", 90, "residential", 0.80, 0.75),
        ("B", "C", 120, "tertiary", 0.55, 0.45),
        ("A", "D", 180, "service", 0.70, 0.85),
        ("D", "C", 80, "residential", 0.85, 0.80),
        # This edge lies outside the demonstration resident catchment.
        ("D", "E", 350, "primary", 0.15, 0.10),
    ]
    for u, v, length, highway, swi, dwi in edges:
        G.add_edge(
            u, v, length_m=length, highway=highway, SWI=swi, DWI=dwi,
            dwi_schedule={"08:00": dwi, "08:05": max(0.0, dwi - 0.05), "default": dwi},
        )
    return G


def main():
    config = yaml.safe_load((ROOT / "configs" / "example_config.yml").read_text())
    data_dir = ROOT / "example_data"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    mobile = pd.read_csv(data_dir / "mobile_pings.csv")
    pois = pd.read_csv(data_dir / "pois.csv")

    all_trips = []
    all_stays = []
    for user_id, group in mobile.groupby("user_id"):
        stays, trips = process_user_trajectory(
            user_id, group, pois,
            eps_m=config["stay_detection"]["eps_m"],
            min_pts=config["stay_detection"]["min_pts"],
            min_duration=config["stay_detection"]["min_duration_min"],
            poi_buffer_m=config["stay_detection"]["poi_buffer_m"],
        )
        stays["user_id"] = user_id
        if not trips.empty:
            trips["trip_id"] = [f"{user_id}_{i}" for i in range(len(trips))]
            all_trips.append(trips)
        all_stays.append(stays)
    stays_df = pd.concat(all_stays, ignore_index=True)
    trips_df = pd.concat(all_trips, ignore_index=True)
    stays_df.to_csv(out_dir / "demo_stays.csv", index=False)
    trips_df.to_csv(out_dir / "demo_trips.csv", index=False)

    routes = pd.read_csv(data_dir / "cached_routes.csv")
    routes["geometry"] = routes["geometry_wkt"].apply(wkt.loads)
    route_checks = []
    for _, trip in trips_df.iterrows():
        route = routes[routes["trip_id"] == trip["trip_id"]].iloc[0]
        # Use only intermediate trajectory pings, excluding the OD endpoints.
        user_pings = mobile[(mobile["user_id"] == trip["user_id"]) &
                            (pd.to_datetime(mobile["timestamp"]) > pd.to_datetime(trip["departure_time"])) &
                            (pd.to_datetime(mobile["timestamp"]) < pd.to_datetime(trip["arrival_time"]))]
        pings = list(user_pings[["longitude", "latitude"]].itertuples(index=False, name=None))
        ok, match_ratio = validate_route_geometry(pings, route["geometry"], threshold_m=config["route_validation"]["baseline_buffer_m"])
        route_checks.append({"trip_id": trip["trip_id"], "valid_route": ok, "match_ratio": match_ratio})
    route_checks_df = pd.DataFrame(route_checks)
    route_checks_df.to_csv(out_dir / "demo_route_checks.csv", index=False)

    # Walking speed sensitivity on synthetic trips
    trips_df["route_length_m"] = trips_df["euclidean_distance_m"]
    wsens = walking_threshold_sensitivity(trips_df, thresholds=[4, 5, 6, 8, 10], distance_col="route_length_m")
    wsens.to_csv(out_dir / "demo_walking_threshold_sensitivity.csv", index=False)

    # Truck GMM stop detection
    truck = pd.read_csv(data_dir / "truck_pings.csv")
    stops = identify_truck_stops_adaptive(truck)
    stops.to_csv(out_dir / "demo_truck_stops.csv", index=False)

    # SWI/DWI examples
    swi_calc = StaticWalkabilityIndex()
    demo_swi_raw = swi_calc.compute_raw_swi({"highway": "residential", "gradient": 1.0, "bldg_density": 0.3,
                                     "has_amenity": True, "has_transit": False, "green_ratio": 0.1,
                                     "industrial_ratio": 0.0, "near_water": False})
    swi_calc.fit_quantile_normalizer([0.2, demo_swi_raw, 0.9])
    demo_swi = float(swi_calc.get_swi([demo_swi_raw])[0])
    dwi_calc = DynamicWalkabilityIndex(config["walkability"]["pcu_weights"])
    ve = dwi_calc.compute_vehicular_exposure([10, 20, 40], [1, 2, 3], [0, 1, 2])
    dwi_calc.fit_quantile_normalizer(ve)
    dwi_values = dwi_calc.get_dwi(ve).tolist()

    # Behavioral model and scenarios
    G = build_demo_graph()
    observed = pd.DataFrame([
        {"origin_node": "A", "dest_node": "C", "departure_time": "2024-09-01 08:00", "time_period": "0800",
         "observed_PE_SWI": 0.75, "observed_PE_DWI": 0.78, "is_home_based": True,
         "is_extra_neighborhood": True, "local_candidates": ["B", "D"],
         "home_node": "A", "catchment_radius_m": 225, "neighborhood_type": "dense"},
        {"origin_node": "A", "dest_node": "B", "departure_time": "2024-09-01 08:10", "time_period": "0800",
         "observed_PE_SWI": 0.80, "observed_PE_DWI": 0.75, "is_home_based": True,
         "is_extra_neighborhood": False, "local_candidates": ["B"],
         "home_node": "A", "catchment_radius_m": 225, "neighborhood_type": "sparse"},
    ])
    calibrator = BehavioralModelCalibration(observed, G)
    params, mse = calibrator.grid_search_parameters(
        sigma_values=np.asarray(config["behavioral_model"]["demo_sigma_values"]),
        beta_values=np.asarray(config["behavioral_model"]["demo_beta_values"]),
    )
    simulator = PolicyScenarioSimulator(G, observed, params["sigma"], params["beta"])
    scenario_frames = [
        simulator.scenario_amenity_localization(alpha_values=[0, 0.5, 1.0]),
        simulator.scenario_built_environment_improvement(alpha_values=[0, 0.5, 1.0]),
        simulator.scenario_traffic_calming(alpha_values=[0, 0.5, 1.0]),
    ]
    scenarios = pd.concat(scenario_frames, ignore_index=True)
    scenarios.to_csv(out_dir / "demo_policy_scenarios.csv", index=False)

    summary = {
        "n_stays": int(len(stays_df)),
        "n_trips": int(len(trips_df)),
        "route_valid_share": float(route_checks_df["valid_route"].mean()),
        "truck_stop_threshold_kmh": float(stops["speed_threshold_kmh"].iloc[0]) if not stops.empty else None,
        "example_raw_swi": demo_swi_raw,
        "example_final_swi_percentile": demo_swi,
        "example_dwi_values": dwi_values,
        "calibrated_params": params,
    }
    (out_dir / "demo_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
