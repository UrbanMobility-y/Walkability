import pandas as pd
from mobility_data_processing import (
    extract_trips_from_stays,
    filter_stays_temporal,
    identify_home_location,
    identify_stays_dbscan,
)
from spatial_utils import haversine_distance_m


def test_dbscan_uses_haversine_and_handles_longitude_compression():
    # Around Beijing latitude, 50 m east corresponds to a larger degree offset than 50/111000.
    lat = 39.9
    lon0 = 116.4
    lon1 = lon0 + 50.0 / (111320.0 * 0.766)  # approx 50 m east at this latitude
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-09-01', periods=4, freq='min'),
        'longitude': [lon0, lon1, lon0, lon1],
        'latitude': [lat, lat, lat, lat],
    })
    stays = identify_stays_dbscan(df, eps_m=60, min_pts=2)
    assert len(stays) == 1
    assert stays.iloc[0]['num_pings'] == 4


def test_filter_default_is_five_minutes():
    stays = pd.DataFrame({'duration_minutes': [4.9, 5.0, 15.0]})
    out = filter_stays_temporal(stays)
    assert out['duration_minutes'].tolist() == [5.0, 15.0]


def test_recurrent_cluster_is_split_into_consecutive_visits():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-09-01 08:00', periods=6, freq='5min'),
        'longitude': [116.4, 116.4, 116.41, 116.41, 116.4, 116.4],
        'latitude': [39.9] * 6,
    })
    stays = identify_stays_dbscan(df, eps_m=60, min_pts=2)
    recurrent = stays[stays['cluster'] == stays.iloc[0]['cluster']]
    assert len(recurrent) == 2


def test_extract_trips_from_stays():
    stays = pd.DataFrame({
        'centroid_lon': [116.4, 116.401],
        'centroid_lat': [39.9, 39.9],
        'start_time': pd.to_datetime(['2024-09-01 08:00', '2024-09-01 08:10']),
        'end_time': pd.to_datetime(['2024-09-01 08:05', '2024-09-01 08:15']),
    })
    trips = extract_trips_from_stays('u1', stays)
    assert len(trips) == 1
    assert trips[0]['euclidean_distance_m'] > 80


def test_home_uses_identified_stay_duration_and_residential_poi():
    records = []
    for day in pd.date_range('2024-09-01', periods=25, freq='D'):
        records.append({
            'cluster': 1,
            'centroid_lon': 116.40,
            'centroid_lat': 39.90,
            'start_time': day + pd.Timedelta(hours=21),
            'end_time': day + pd.Timedelta(hours=23),
            'poi_category': 'residential',
        })
        records.append({
            'cluster': 2,
            'centroid_lon': 116.45,
            'centroid_lat': 39.95,
            'start_time': day + pd.Timedelta(hours=21),
            'end_time': day + pd.Timedelta(hours=22),
            'poi_category': 'residential',
        })
    home = identify_home_location(pd.DataFrame(records))
    assert home['is_valid'] is True
    assert home['cluster'] == 1
    assert home['nights_present'] == 25
    assert home['nighttime_duration_minutes'] == 3000.0
    assert home['weekend_nighttime_duration_minutes'] > 0


def test_home_rejects_nonresidential_candidate_and_supports_weekend_rule():
    rows = []
    for day in pd.date_range('2024-09-01', periods=25, freq='D'):
        rows.extend([
            {
                'cluster': 1, 'centroid_lon': 116.40, 'centroid_lat': 39.90,
                'start_time': day + pd.Timedelta(hours=21),
                'end_time': day + pd.Timedelta(hours=23),
                'poi_category': 'office',
            },
            {
                'cluster': 2, 'centroid_lon': 116.41, 'centroid_lat': 39.91,
                'start_time': day + pd.Timedelta(hours=21),
                'end_time': day + pd.Timedelta(hours=22),
                'poi_category': 'residential',
            },
        ])
    home = identify_home_location(
        pd.DataFrame(rows),
        weekend_validator=lambda candidate: candidate['weekend_nights_present'] >= 1,
    )
    assert home['cluster'] == 2
    assert home['passes_residential_check']
    assert home['passes_weekend_check']
