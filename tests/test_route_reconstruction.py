from shapely.geometry import LineString
from route_reconstruction import validate_route_geometry, route_overlap_ratio


def test_route_validation_metric_buffer():
    route = LineString([(116.4, 39.9), (116.401, 39.9)])
    pings = [(116.4002, 39.90001), (116.4005, 39.89999), (116.4008, 39.90002)]
    ok, ratio = validate_route_geometry(pings, route, threshold_m=20, ratio=0.8)
    assert ok
    assert ratio == 1.0


def test_route_overlap_ratio():
    a = LineString([(116.4, 39.9), (116.401, 39.9)])
    b = LineString([(116.4, 39.9), (116.401, 39.9)])
    assert route_overlap_ratio(a, b) > 0.95
