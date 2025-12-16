# Two Dimensions of Walkability Diverge Across Urban Forms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Computational framework and analysis code for quantifying the dual dimensions of urban walkability: static built environment quality and dynamic traffic exposure.

---

## 📁 Repository Structure

```
.
├── mobility_data_processing.py    # [Data]   Mobile phone trajectory cleaning & stay detection
├── truck_data_processing.py       # [Data]   Truck GPS processing with adaptive stop detection
├── transport_mode_inference.py    # [Algo]   Mode inference using XGBoost & Travel Time Ratios
├── urban_morphology.py            # [Env]    Urban form classification
├── route_reconstruction.py        # [Algo]   Route reconstruction via API & High-precision Map Matching
├── walkability_indices.py         # [Metric] Static (SWI) & Dynamic (DWI) index calculation
├── behavioral_modeling_policy.py  # [Sim]    Agent-based routing & Policy scenario simulation
└── README.md
```

---

## 🔧 Core Modules

**`mobility_data_processing.py`** — Preprocess anonymized mobile phone data to identify user stay points and infer primary residence locations.

**`truck_data_processing.py`** — Clean high-frequency GPS trajectories and identify logistics stops using an adaptive approach.

**`urban_morphology.py`** — Classify cities into four urban forms (Radial Monocentric, Concentric Monocentric, Clustered Polycentric, Dispersed Polycentric).

**`transport_mode_inference.py`** — Infer travel modes (Walking/Private/Public) using multi-dimensional features (trip distance, duration, travel time ratios vs. routing API benchmarks).

**`route_reconstruction.py`** — Reconstruct precise street-level trajectories for pedestrians and vehicles using mapping APIs and map-matching algorithms.

**`walkability_indices.py`** — Compute static and dynamic walkability metrics across three spatial scales—street segments (SWI/DWI), 15-minute neighborhood catchments (Home Indices), and time-weighted individual trip trajectories (Pedestrian Experience).

**`behavioral_modeling_policy.py`** — Implement the agent-based simulation model for policy scenarios within the 15-minute city framework.

---

## ⚙️ Configuration

### API Keys

The `route_reconstruction.py` and `transport_mode_inference.py` modules require an [AutoNavi (Amap)](https://lbs.amap.com/) API key to fetch routing data.

```python
from route_reconstruction import RouteReconstructor

reconstructor = RouteReconstructor(api_key="YOUR_AMAP_KEY")
```

> **Note:** You can obtain an API key by registering at the [Amap Open Platform](https://lbs.amap.com/).

---

## 🚀 Quick Start

The following examples demonstrate the core workflow: from raw data mining to policy simulation.

### 1. Identify Home Locations from GPS Data

Extract user home locations from raw mobile phone signaling data based on nighttime presence.

```python
import pandas as pd
from mobility_data_processing import identify_home_location

# Mock raw trajectory data
trajectory_df = pd.DataFrame({
    'timestamp': pd.date_range(start='2024-09-01 22:00', periods=5, freq='H'),
    'latitude': [39.90, 39.90, 39.90, 39.90, 39.91],
    'longitude': [116.40, 116.40, 116.40, 116.40, 116.41]
})

# Identify home location (nighttime: 21:00-06:00)
home_info = identify_home_location(trajectory_df, min_days_present=1)
print(f"Home Coordinates: ({home_info['longitude']:.4f}, {home_info['latitude']:.4f})")
```

### 2. Classify Urban Form

Identify population centers and classify the city structure (e.g., Polycentric vs. Monocentric).

```python
from urban_morphology import UrbanMorphologyClassifier

# Initialize classifier
classifier = UrbanMorphologyClassifier(grid_size_degrees=0.005)

# Assuming 'population_grid' is a GeoDataFrame with smoothed population
centers = classifier.identify_population_centers(population_grid)
city_type = classifier.classify_polycentric_subtype(centers, population_grid)

print(f"Identified {len(centers)} centers. City Type: {city_type}")
```

### 3. Reconstruct & Match Routes

Reconstruct detailed walking paths via API and map them to OSM segments with metric precision.

```python
from route_reconstruction import batch_reconstruct_trips, segment_routes_optimized

# Input OD data
trips = pd.DataFrame([{
    'trip_id': 'T001', 
    'o_lon': 116.39, 'o_lat': 39.90, 
    'd_lon': 116.41, 'd_lat': 39.92
}])

# 1. Reconstruct Route (Requires AMap API Key)
# Returns Shapely LineString geometries
routes_df = batch_reconstruct_trips(trips, gps_data_df, api_key="YOUR_KEY", mode='pedestrian')

# 2. Map Matching (Spatial Join with OSM Network)
# Calculates precise overlap length in meters (projection corrected)
segments = segment_routes_optimized(routes_df, osm_network_gdf, buffer_size_m=20)

print(segments[['trip_id', 'osm_segment_id', 'overlap_length_m']].head())
```

### 4. Calculate Walkability Indices

Compute the Static Walkability Index (SWI) for a street segment.

```python
from walkability_indices import StaticWalkabilityIndex

# Initialize calculator
swi_calc = StaticWalkabilityIndex()

# Define segment attributes
segment_data = {
    'highway': 'residential',
    'gradient': 1.5,
    'bldg_density': 0.4,
    'has_amenity': True,
    'green_ratio': 0.6
}

# Compute score
score = swi_calc.compute_swi(segment_data)
print(f"Segment SWI: {score:.2f}")
```

### 5. Policy Simulation (Agent-Based)

Calibrate behavioral parameters and simulate the "15-Minute City" Amenity Localization scenario.

```python
from behavioral_modeling_policy import BehavioralModelCalibration, PolicyScenarioSimulator

# 1. Calibrate Model (Find optimal beta and sigma)
calibrator = BehavioralModelCalibration(observed_trips, street_network)
params, _ = calibrator.grid_search_parameters(sample_size=1000)
print(f"Calibrated Parameters: {params}")

# 2. Run 'Amenity Localization' Scenario
# Simulate redirecting 30% of long-distance trips to local neighborhoods
simulator = PolicyScenarioSimulator(street_network, observed_trips, params)
results = simulator.scenario_amenity_localization(alpha_values=[0.0, 0.3])

print(results[['alpha', 'PE_Composite', 'PE_SWI', 'PE_DWI']])
```

---

## 📄 License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.




























# Two Dimensions of Walkability Diverge Across Urban Forms

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Computational framework and analysis code for quantifying the dual dimensions of urban walkability: static built environment quality and dynamic traffic exposure.

---

## 📁 Repository Structure

```
.
├── mobility_data_processing.py    # [Data]   Mobile phone trajectory cleaning & stay detection
├── truck_data_processing.py       # [Data]   Truck GPS processing with adaptive stop detection
├── transport_mode_inference.py    # [Algo]   Mode inference using XGBoost & Travel Time Ratios
├── urban_morphology.py            # [Env]    Urban form classification
├── route_reconstruction.py        # [Algo]   Route reconstruction via API & High-precision Map Matching
├── walkability_indices.py         # [Metric] Static (SWI) & Dynamic (DWI) index calculation
├── behavioral_modeling_policy.py  # [Sim]    Agent-based routing & Policy scenario simulation
└── README.md
```

---

## 🔧 Core Modules

**`mobility_data_processing.py`** — Preprocess anonymized mobile phone data to identify user stay points and infer primary residence locations.

**`truck_data_processing.py`** — Clean high-frequency GPS trajectories and identify logistics stops using an adaptive approach.

**`urban_morphology.py`** — Classify cities into four urban forms (Radial Monocentric, Concentric Monocentric, Clustered Polycentric, Dispersed Polycentric).

**`transport_mode_inference.py`** — Infer travel modes (Walking/Private/Public) using multi-dimensional features (trip distance, duration, travel time ratios vs. routing API benchmarks).

**`route_reconstruction.py`** — Reconstruct precise street-level trajectories for pedestrians and vehicles using mapping APIs and map-matching algorithms.

**`walkability_indices.py`** — Compute static and dynamic walkability metrics across three spatial scales—street segments (SWI/DWI), 15-minute neighborhood catchments (Home Indices), and time-weighted individual trip trajectories (Pedestrian Experience).

**`behavioral_modeling_policy.py`** — Implement the agent-based simulation model for policy scenarios within the 15-minute city framework.

---

## ⚙️ Configuration

### API Keys

The `route_reconstruction.py` and `transport_mode_inference.py` modules require an [AutoNavi (Amap)](https://lbs.amap.com/) API key to fetch routing data.

```python
from route_reconstruction import RouteReconstructor

reconstructor = RouteReconstructor(api_key="YOUR_AMAP_KEY")
```

> **Note:** You can obtain an API key by registering at the [Amap Open Platform](https://lbs.amap.com/).

---

## 🚀 Quick Start

### 1. Calculate Walkability Indices

Compute the Static Walkability Index (SWI) for a street segment:

```python
from walkability_indices import StaticWalkabilityIndex

# Initialize calculator
swi_calc = StaticWalkabilityIndex()

# Define segment attributes
segment_data = {
    'highway': 'residential',
    'gradient': 1.5,
    'bldg_density': 0.4,
    'has_amenity': True,
    'green_ratio': 0.6
}

# Compute score
score = swi_calc.compute_swi(segment_data)
print(f"Segment SWI: {score:.2f}")
```

### 2. Run Policy Simulation

Simulate the impact of the Traffic Calming scenario on pedestrian experience:

```python
from behavioral_modeling_policy import PolicyScenarioSimulator

# Initialize simulator with calibrated parameters
params = {'sigma': 0.55, 'beta': 1.8}
simulator = PolicyScenarioSimulator(street_network, observed_trips, params)

# Run scenario
results = simulator.scenario_traffic_calming(alpha_values=np.arange(0, 1.1, 0.1))
print(results)
```

---

## 📄 License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.


