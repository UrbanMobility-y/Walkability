#Overview
This repository contains the computational framework and analysis code for the paper "Two dimensions of walkability diverge across urban forms".

#Repository Structure
The codebase is organized into 7 modules, each handling a specific aspect of the pipeline:
.
├── mobility_data_processing.py    # [Data] Mobile phone trajectory cleaning & stay detection
├── truck_data_processing.py       # [Data] Truck GPS processing with adaptive stop detection
├── transport_mode_inference.py    # [Algo] Mode inference using XGBoost & Travel Time Ratios
├── urban_morphology.py            # [Env]  Urban form classification
├── route_reconstruction.py        # [Algo] Route reconstruction via API & High-precision Map Matching
├── walkability_indices.py         # [Metric] Static (SWI) & Dynamic (DWI) index calculation
├── behavioral_modeling_policy.py  # [Sim]  Agent-based routing & Policy scenario simulation
└── README.md

#Core Modules
mobility_data_processing.py: Preprocess anonymized mobile phone data to identify user stay points and infer primary residence locations.
truck_data_processing.py: Clean high-frequency GPS trajectories and identifies logistics stops using adaptive approach.
urban_morphology.py: Classify cities into four urban forms (Radial Monocentric, Concentric Monocentric, Clustered Polycentric, Dispersed Polycentric).
transport_mode_inference.py: Infer travel modes (Walking/Private/Public) using multi-dimensional features (trip distance, duration, travel time ratios vs. routing API benchmarks).
route_reconstruction.py: Reconstruct precise street-level trajectories for pedestrians and vehicles using mapping APIs and map-matching algorithms.
walkability_indices.py: Compute static and dynamic walkability metrics across three spatial scales——street segments (SWI/DWI), 15-minute neighborhood catchments (Home_Indices), and time-weighted individual trip trajectories (Pedestrian Experience PE).
behavioral_modeling_policy.py: Implement the agent-based simulation model for policy scenarios within the 15-minute city framework.

#Quick Start

Usage Examples

1. Calculate Walkability Indices: Compute the Static Walkability Index (SWI) for a street segment.

from walkability_indices import StaticWalkabilityIndex
swi_calc = StaticWalkabilityIndex()# Initialize calculator
segment_data = {
    'highway': 'residential',
    'gradient': 1.5,
    'bldg_density': 0.4,
    'has_amenity': True,
    'green_ratio': 0.6}# Define segment attributes
score = swi_calc.compute_swi(segment_data)
print(f"Segment SWI: {score:.2f}")

2. Run Policy Simulation: Simulate the impact of Traffic Calming scenario on pedestrian experience.

from behavioral_modeling_policy import PolicyScenarioSimulator
params = {'sigma': 0.55, 'beta': 1.8}# Initialize simulator with calibrated parameters
simulator = PolicyScenarioSimulator(street_network, observed_trips, params)
results = simulator.scenario_traffic_calming(alpha_values=[0.0, 0.3])
print(results)

#Configuration
API Keys: The route_reconstruction.py and transport_mode_inference.py modules require an AutoNavi (Amap) API Key to fetch routing data.
reconstructor = RouteReconstructor(api_key="YOUR_AMAP_KEY")

