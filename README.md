# Experienced walkability and the paradox of urban density in Chinese cities

This repository is the public code companion focused on methods for the study
**“Experienced walkability and the paradox of urban density in Chinese
cities.”** It implements the analytical framework on synthetic/example data
and includes portable notebooks for the privacy safe source data release.

It is **not** a release of the restricted production pipeline used to process
82 cities and 1.50 billion reconstructed likely walking routes. The raw mobile
phone records, heavy truck trajectories, complete five-minute segment exposure
fields, full route cache and individual simulation inputs cannot be shared.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for a table describing each component.

## What the code implements

- Mobile ping quality control, monthly DBSCAN activity clusters, visit level
  dwell filtering, POI matching, trip OD extraction and home inference.
- Three class XGBoost mode inference, high confidence active mode screening and
  likely walking refinement at 6 km/h.
- Cached AMap route handling, route/ping validation and route overlap
  sensitivity utilities.
- Heavy truck preprocessing, adaptive GMM stop detection and OD processing.
- Adapted NetAScore raw SWI and the **city specific ECDF conversion required for
  final SWI percentile positions**.
- Five-minute vehicular exposure from car/bus/truck density, city specific DWI
  percentiles, conservative node level DCI and crossing dwell rules.
- Circuity adjusted home centered 15-minute catchments and length weighted
  neighborhood aggregation.
- Time aware route level `PE_SWI`, `PE_DWI` and model derived composite
  experience.
- Calibration sampling by distance, time and spatial origin; grid search over sigma =
  0–1 by 0.05 and beta = 0–5 by 0.1, plus the 5% MSE sensitivity envelope.
- Three controlled policy scenarios at intensities 0–1 by 0.1.

SWI and DWI perturbations require an explicit residential catchment and are
applied only to street segments within that catchment. Amenity localization
redirects only home based extra neighborhood trips to supplied local
candidates. All scenarios hold the underlying raw vehicle field fixed and do
not model system feedback.

## Repository structure

```text
.
├── mobility_data_processing.py       # pings, stays, ODs and homes
├── transport_mode_inference.py       # XGBoost and likely walking refinement
├── route_reconstruction.py           # AMap/cached routes and validation
├── truck_data_processing.py          # heavy truck processing
├── walkability_indices.py            # raw/final SWI, VE, DWI, DCI and PE
├── neighborhood_walkability.py       # 15-minute catchments and aggregation
├── classify_urban_forms.py           # operational population form typology
├── behavioral_modeling_policy.py     # calibration and policy experiments
├── spatial_utils.py                  # metric distance/projection helpers
├── configs/example_config.yml        # paper parameters and demo overrides
├── configs/urban_form_config.json    # frozen depth 7 classification tree
├── example_data/                     # synthetic analogues of restricted schemas
├── scripts/run_demo_pipeline.py      # runnable method demonstration
├── Result replication/               # portable notebooks with saved figure outputs
├── tests/                             # unit and boundary tests
└── REPRODUCIBILITY.md
```

## Installation

With conda:

```bash
conda env create -f environment.yml
conda activate walkability-reproducible
```

Or with a virtual environment and pip:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run the synthetic method demonstration

From the repository root:

```bash
python scripts/run_demo_pipeline.py
python -m pytest -q
```

The demo is deterministic, does not call AMap and writes to `outputs/`:

```text
demo_stays.csv
demo_trips.csv
demo_route_checks.csv
demo_walking_threshold_sensitivity.csv
demo_truck_stops.csv
demo_policy_scenarios.csv
demo_summary.json
```

The small calibration grid in the demo is a runtime convenience. The full
manuscript parameter ranges and calibration sample size are recorded in
`configs/example_config.yml`. The sigma and beta ranges are also the defaults
in `BehavioralModelCalibration.grid_search_parameters`.

## Reproduce figures from the shareable source data

Each notebook contains two complementary records. The first section is a
portable implementation that uses the shareable source data described below.
The second section retains the authors' workstation code, absolute paths,
execution counts and saved outputs. Keeping the executed code beside its output
allows readers to verify directly on GitHub how the displayed figure results
were generated.

Download the released data from
[Zenodo (DOI: 10.5281/zenodo.21902140)](https://zenodo.org/records/21902140)
and place the `source data` directory next to this repository:

```text
Code and Data/
├── Github code/
└── source data/
```

Start Jupyter from the repository root and run the notebooks in `Result
replication` in order:

```bash
jupyter notebook
```

If the data are elsewhere, set `WALKABILITY_SOURCE_DATA` before starting
Jupyter. For example, in PowerShell:

```powershell
$env:WALKABILITY_SOURCE_DATA = "D:\data\source data"
jupyter notebook
```

When executed, the portable cells regenerate figures from released daily street
and neighborhood indices, resident level aggregate curves and policy summaries
for four cities. They do not reconstruct unavailable five-minute production
inputs. The portable city location panel in Figure 1 uses released coordinates
because the production cartographic boundary files are not included, so its
basemap styling differs from the authors' executed workstation output.

## Run the urban form classifier

The default configuration contains the frozen numerical multiclass decision
tree with maximum depth 7 used for the 82 city classification. It is loaded
automatically when `--config` is omitted:

```bash
python classify_urban_forms.py \
  --input-dir "../source data/neighborhood-level walkability" \
  --output-dir outputs/urban_forms
```

## Key input schemas

### Mobile phone pings

Required: `user_id`, `timestamp`, `longitude`, `latitude`, `city_code`.

Home identification operates on identified visit level stays rather than raw
pings. Required columns are `cluster`, `centroid_lon`, `centroid_lat`,
`start_time`, `end_time`, plus either `poi_category` or
`is_residential_poi`. It selects residential candidates observed on at least
25 nights by cumulative duration within 21:00 to 06:00. Weekend nighttime
duration is reported, and a study specific validator can be supplied because
the paper does not report a numerical weekend cutoff.

### Cached routes

Required: `trip_id`, `geometry` (Shapely `LineString` or parsed WKT), `mode`.

### Truck pings

Required: `truck_id`, `timestamp`, `longitude`, `latitude`, `speed_kmh`.

### Street graph

Edges use `length_m`, final within city percentile `SWI`, and `DWI`,
`DWI_HHMM`/`DWI_HH:MM`, or `dwi_schedule`. Nodes may use `DCI`/`dci_schedule`
and `dwell_time_s`. Metric `x`, `y` node coordinates or Shapely edge geometry
allow the public code to infer home catchments.

For MultiGraph and MultiDiGraph networks, optimized routes retain the selected
edge key for each segment so route experience is evaluated on the same parallel
edge used by the optimizer.

### Policy trips

Required: `origin_node`, `dest_node`, `departure_time`. SWI/DWI scenarios also
require either `catchment_edges`, or `home_node`/`home_x`/`home_y` plus
`catchment_radius_m`. Amenity localization additionally uses
`is_extra_neighborhood`, optional `is_home_based`, and `local_candidates`.
`neighborhood_type` enables dense/sparse stratification.

## Interpretation

Final SWI and DWI values are within city empirical percentile positions. A
value of 0.9 denotes a condition more favorable than approximately 90% of the
corresponding city reference distribution; it does not equate the underlying
built environment and traffic quantities. In the SWI and DWI scenarios, alpha
values perturb the corresponding normalized index toward 1 and should not be
interpreted as percentages of physical improvement or traffic reduction. In
the amenity localization scenario, `alpha_POI` denotes the proportion of
eligible home based extra neighborhood trips redirected to local destinations.

## License

MIT. See [LICENSE](LICENSE).
