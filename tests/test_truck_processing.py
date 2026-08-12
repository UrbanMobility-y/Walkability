import numpy as np
import pandas as pd
from truck_data_processing import estimate_stop_speed_threshold_gmm, identify_truck_stops_adaptive


def test_gmm_threshold_between_stop_and_move_modes():
    rng = np.random.default_rng(42)
    speeds = np.r_[rng.normal(0.8, 0.2, 200), rng.normal(35, 5, 200)]
    th = estimate_stop_speed_threshold_gmm(speeds)
    assert 0.5 <= th <= 10.0


def test_identify_truck_stops_uses_gmm_column():
    df = pd.DataFrame({
        'truck_id': ['t1'] * 40,
        'timestamp': pd.date_range('2024-09-01', periods=40, freq='30s'),
        'longitude': np.linspace(116.3, 116.31, 40),
        'latitude': [39.8] * 40,
        'speed_kmh': [0.5] * 10 + [30] * 20 + [0.6] * 10,
    })
    stops = identify_truck_stops_adaptive(df)
    assert not stops.empty
    assert 'speed_threshold_kmh' in stops.columns
