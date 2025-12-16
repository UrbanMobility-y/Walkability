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


