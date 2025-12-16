"""
Urban Morphology and Built Environment Module

"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import distance, ConvexHull
from scipy.optimize import curve_fit
from shapely.geometry import Point, Polygon
import warnings

# Suppress warnings for curve_fit convergence in noisy data
warnings.filterwarnings('ignore', category=RuntimeWarning)

class UrbanMorphologyClassifier:
    """Classifies urban forms based on population distribution patterns (Note 2.3)"""

    def __init__(self, grid_size_degrees=0.005, smoothing_sigma_km=1.0):
        self.grid_size_degrees = grid_size_degrees
        self.smoothing_sigma_km = smoothing_sigma_km

    def create_population_grid(self, users_home_locations, city_boundary):
        """Creates 0.005 degree grid with population counts."""
        minx, miny, maxx, maxy = city_boundary.bounds

        # Vectorized grid creation
        lons = np.arange(minx, maxx, self.grid_size_degrees)
        lats = np.arange(miny, maxy, self.grid_size_degrees)

        # Create polygons
        # Note: Using meshgrid for efficiency
        xx, yy = np.meshgrid(lons, lats)
        cells = []
        for x, y in zip(xx.flatten(), yy.flatten()):
            cells.append(Polygon([
                (x, y), (x + self.grid_size_degrees, y),
                (x + self.grid_size_degrees, y + self.grid_size_degrees),
                (x, y + self.grid_size_degrees)
            ]))

        grid_gdf = gpd.GeoDataFrame(geometry=cells)

        # Spatial join to count points (much faster than looping)
        users_gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(users_home_locations.longitude, 
                                      users_home_locations.latitude)
        )

        # Count points in each cell
        joined = gpd.sjoin(users_gdf, grid_gdf, how="inner", predicate="within")
        counts = joined.index_right.value_counts()

        grid_gdf['population'] = 0
        grid_gdf.loc[counts.index, 'population'] = counts.values

        # Add centroids for processing
        grid_gdf['longitude'] = grid_gdf.centroid.x
        grid_gdf['latitude'] = grid_gdf.centroid.y

        return grid_gdf

    def smooth_population_surface(self, population_grid):
        """Applies Gaussian smoothing (Eq. 1)."""
        # Note: This is an O(N^2) operation. For large cities, FFT convolution is preferred,
        # but explicit summation is retained here for strict adherence to the formula.

        coords = population_grid[['longitude', 'latitude']].values
        pops = population_grid['population'].values
        sigma_deg = self.smoothing_sigma_km / 111.0

        smoothed = []
        for i, point in enumerate(coords):
            dists_sq = np.sum((coords - point)**2, axis=1)
            # Threshold optimization: only compute for nearby points (e.g., < 3 sigma)
            # to speed up without losing accuracy
            mask = dists_sq < (3 * sigma_deg)**2

            if np.sum(mask) == 0:
                smoothed.append(0)
                continue

            weights = np.exp(-dists_sq[mask] / (2 * sigma_deg**2))
            weighted_sum = np.sum(pops[mask] * weights)
            sum_weights = np.sum(weights)

            smoothed.append(weighted_sum / sum_weights if sum_weights > 0 else 0)

        population_grid['smoothed_population'] = smoothed
        return population_grid

    def identify_population_centers(self, grid, significance_threshold=0.45):
        """Identifies centers using local maxima and density thresholds (Step 2)."""

        # 1. Identify Local Maxima (1km radius)
        # 1km approx 2 grid cells (0.005 deg * 2 * 111 = 1.1km)
        # Using a simple 3x3 window check on the grid logic if structured,
        # or distance-based check for unstructured GDF.

        candidates = []
        coords = grid[['longitude', 'latitude']].values
        pops = grid['smoothed_population'].values
        radius_deg = 1.0 / 111.0

        # Pre-filter: must be a peak among immediate neighbors
        for i in range(len(grid)):
            p = pops[i]
            c = coords[i]

            # Distance to all other points (optimization: use KDTree in prod)
            dists = np.sqrt(np.sum((coords - c)**2, axis=1))
            neighbors_mask = (dists <= radius_deg) & (dists > 0)

            if np.all(p >= pops[neighbors_mask]):
                candidates.append(i)

        # 2. Global Max (Primary Center)
        cand_df = grid.iloc[candidates].copy()
        if cand_df.empty: return [] # Handle edge case

        primary_idx = cand_df['smoothed_population'].idxmax()
        primary_pop = cand_df.loc[primary_idx, 'smoothed_population']
        primary_center = cand_df.loc[primary_idx]

        final_centers = [primary_center]

        # 3. Secondary Centers (Threshold & Separation)
        # Sort candidates by population desc
        cand_df = cand_df.sort_values('smoothed_population', ascending=False)

        min_sep_deg = 2.0 / 111.0

        for idx, row in cand_df.iterrows():
            if idx == primary_idx: continue

            # Density Threshold
            if row['smoothed_population'] < significance_threshold * primary_pop:
                continue

            # Separation Check
            is_far_enough = True
            for existing in final_centers:
                d = np.sqrt((row.longitude - existing.longitude)**2 + 
                            (row.latitude - existing.latitude)**2)
                if d < min_sep_deg:
                    is_far_enough = False
                    break

            if is_far_enough:
                final_centers.append(row)

        return pd.DataFrame(final_centers)

    def classify_monocentric_subtype(self, grid, primary_center):
        """
        Distinguishes Concentric vs Radial (Step 4).
        Corrected to use scipy.optimize.curve_fit.
        """
        center_lon = primary_center['longitude']
        center_lat = primary_center['latitude']

        # Data preparation
        grid['dist_km'] = np.sqrt(
            (grid.longitude - center_lon)**2 + (grid.latitude - center_lat)**2
        ) * 111.0

        data = grid[grid['smoothed_population'] > 0]
        x = data['dist_km'].values
        y = data['smoothed_population'].values

        # 1. Angular Variance
        # Calculate CV for rings
        cv_list = []
        for r in range(1, int(x.max()) + 1):
            ring = data[(data.dist_km >= r-0.5) & (data.dist_km < r+0.5)]
            if len(ring) > 5:
                cv_list.append(ring['smoothed_population'].std() / ring['smoothed_population'].mean())
        mean_cv = np.mean(cv_list) if cv_list else 1.0

        # 2. Density Decay Models
        def exp_func(d, a, b): return a * np.exp(-b * d)
        def pow_func(d, a, b): return a * np.power(d, -b)

        def get_r2(func, x_vals, y_vals):
            try:
                # Provide reasonable bounds/guesses to aid convergence
                popt, _ = curve_fit(func, x_vals, y_vals, 
                                  p0=[y_vals.max(), 0.1], maxfev=2000)
                y_pred = func(x_vals, *popt)
                ss_res = np.sum((y_vals - y_pred)**2)
                ss_tot = np.sum((y_vals - np.mean(y_vals))**2)
                return 1 - (ss_res / ss_tot)
            except:
                return -np.inf

        # Filter x > 0.1 for power law stability
        mask = x > 0.1
        r2_exp = get_r2(exp_func, x[mask], y[mask])
        r2_pow = get_r2(pow_func, x[mask], y[mask])

        if mean_cv <= 0.5 and r2_exp > r2_pow:
            return 'concentric_monocentric'
        return 'radial_monocentric'

    def classify_polycentric_subtype(self, centers, grid):
        """
        Distinguishes Clustered vs Dispersed (Step 5).
        Corrected to calculate Spatial Concentration (SC) properly.
        """
        n = len(centers)
        points = centers[['longitude', 'latitude']].values

        # 1. ANN Ratio
        # Observed mean nearest neighbor distance
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nbrs.kneighbors(points)
        d_obs = np.mean(distances[:, 1])

        # Expected distance (A = Convex Hull of city or total grid area)
        # Note 2.3 Eq 5 uses 'study area'. 
        # Using grid bounding box area as proxy for 'A'
        minx, miny, maxx, maxy = grid.geometry.total_bounds
        area_deg2 = (maxx-minx) * (maxy-miny)
        area_km2 = area_deg2 * (111**2)
        d_exp = 0.5 * np.sqrt(area_km2 / n)

        R = d_obs / d_exp # ANN Ratio

        # 2. Spatial Concentration (SC)
        # SC = Pop in Hull / Total Pop
        sc = 0.0
        if n >= 3:
            hull = ConvexHull(points)
            hull_poly = Polygon(points[hull.vertices])

            # Find population within hull
            # Using spatial index or simple check
            in_hull = grid[grid.geometry.centroid.within(hull_poly)]
            pop_in_hull = in_hull['population'].sum()
            total_pop = grid['population'].sum()

            sc = pop_in_hull / total_pop if total_pop > 0 else 0

        # Classification Logic
        if 2 <= n <= 3 and R <= 1.0 and sc >= 0.5:
            return 'clustered_polycentric'
        elif n >= 4 and R > 1.0 and sc < 0.5:
            return 'dispersed_polycentric'

        # Fallback based on center count if metrics ambiguous
        return 'clustered_polycentric' if n <= 3 else 'dispersed_polycentric'


class BuiltEnvironmentCharacterizer:
    """
    Characterizes physical attributes using Spatial Indexing for performance.
    (Ref: Note 2.2)
    """

    def _compute_ratio_in_buffer(self, segments, target_gdf, buffer_m):
        """Generic function to compute area ratios using spatial index."""
        # Buffer segments
        buffer_deg = buffer_m / 111000.0
        buffered_segments = segments.copy()
        buffered_segments['geometry'] = segments.geometry.buffer(buffer_deg)

        # Ensure spatial index exists
        if target_gdf.sindex is None:
            _ = target_gdf.sindex

        ratios = []

        # Iterate efficiently
        # Note: sjoin is faster than looping intersects
        joined = gpd.sjoin(buffered_segments, target_gdf, how='left', predicate='intersects')

        # Group by segment index to handle multiple overlaps
        grouped = joined.groupby(joined.index)

        for idx in segments.index:
            if idx not in grouped.groups:
                ratios.append(0.0)
                continue

            # Get targets that intersect this segment's buffer
            # Note: sjoin gives candidate matches based on bounding box (usually), 
            # or precise depending on implementation. 
            # For exact area calculation, we need intersection geometry.

            # Fetch the actual geometries from target_gdf using the indices from join
            match_indices = grouped.get_group(idx)['index_right']
            if match_indices.isnull().all():
                ratios.append(0.0)
                continue

            matches = target_gdf.loc[match_indices.dropna()]

            # Calculate intersection area
            # Clip matches to buffer
            buf_geom = buffered_segments.loc[idx, 'geometry']
            intersections = matches.geometry.intersection(buf_geom)

            total_target_area = intersections.area.sum()
            buffer_area = buf_geom.area

            ratios.append(total_target_area / buffer_area if buffer_area > 0 else 0)

        return ratios

    def compute_building_density(self, buildings, segments):
        return self._compute_ratio_in_buffer(segments, buildings, 50)

    def compute_greenness_ratio(self, land_use, segments):
        green = land_use[land_use['category'].isin(['park', 'vegetation', 'forest'])]
        return self._compute_ratio_in_buffer(segments, green, 50)

    def compute_industrial_ratio(self, land_use, segments):
        ind = land_use[land_use['category'].isin(['industrial', 'warehouse'])]
        return self._compute_ratio_in_buffer(segments, ind, 50)

    def compute_amenity_access(self, pois, segments):
        """Binary check for amenities within 50m."""
        buffer_deg = 50 / 111000.0
        buffered = segments.geometry.buffer(buffer_deg)

        cats = ['shop', 'restaurant', 'school', 'hospital']
        target_pois = pois[pois['category'].isin(cats)]

        # Spatial join check existence
        # Create temp GDF for buffers to join
        buf_gdf = gpd.GeoDataFrame(geometry=buffered, index=segments.index)
        joined = gpd.sjoin(buf_gdf, target_pois, how='inner', predicate='intersects')

        has_amenity = segments.index.isin(joined.index)
        return has_amenity