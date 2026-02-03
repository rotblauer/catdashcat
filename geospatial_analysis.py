"""
Geospatial Analysis Module for CatDash
======================================
Scalable geospatial analysis methods for millions of GPS tracking points.

Methods include:
1. Spatial clustering (DBSCAN, HDBSCAN, K-means on geohash)
2. Hotspot analysis (Kernel Density Estimation)
3. Movement pattern analysis (Stay point detection, trajectory segmentation)
4. Temporal patterns (Time-of-day, day-of-week heatmaps)
5. Home range estimation (Minimum Convex Polygon, Kernel Density)
6. Spatial autocorrelation (Moran's I)
7. Grid-based aggregation (H3, geohash)
8. Trajectory similarity (Frechet distance sampling)
"""

import argparse
import gzip
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from sklearn.neighbors import BallTree
from collections import defaultdict
import json

# Constants
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def haversine_vectorized(lats1, lons1, lats2, lons2):
    """Vectorized haversine distance in kilometers."""
    lats1, lons1 = np.radians(lats1), np.radians(lons1)
    lats2, lons2 = np.radians(lats2), np.radians(lons2)
    dlat = lats2 - lats1
    dlon = lons2 - lons1
    a = np.sin(dlat/2)**2 + np.cos(lats1) * np.cos(lats2) * np.sin(dlon/2)**2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def encode_geohash(lat: float, lon: float, precision: int = 7) -> str:
    """Encode lat/lon to geohash string."""
    BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True

    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if lon > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if lat > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash.append(BASE32[ch])
            bit = 0
            ch = 0
    return ''.join(geohash)


class SpatialClustering:
    """Efficient spatial clustering for large datasets."""

    @staticmethod
    def dbscan_haversine(df: pd.DataFrame, eps_km: float = 0.1,
                         min_samples: int = 5, sample_frac: float = 1.0) -> np.ndarray:
        """
        DBSCAN clustering using haversine distance.

        Args:
            df: DataFrame with 'lat' and 'lon' columns
            eps_km: Maximum distance between points in km
            min_samples: Minimum points to form a cluster
            sample_frac: Fraction to sample for very large datasets
        """
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)

        coords = np.radians(df[['lat', 'lon']].values)

        # Use BallTree with haversine metric for efficiency
        tree = BallTree(coords, metric='haversine')
        eps_rad = eps_km / EARTH_RADIUS_KM

        dbscan = DBSCAN(eps=eps_rad, min_samples=min_samples,
                        metric='precomputed', n_jobs=-1)

        # Get sparse distance matrix
        distances = tree.query_radius(coords, r=eps_rad, return_distance=True)

        # Build sparse distance matrix
        from scipy.sparse import lil_matrix
        n = len(coords)
        dist_matrix = lil_matrix((n, n))
        for i, (indices, dists) in enumerate(zip(*distances)):
            for j, d in zip(indices, dists):
                dist_matrix[i, j] = d

        labels = dbscan.fit_predict(dist_matrix.tocsr())
        return labels

    @staticmethod
    def geohash_kmeans(df: pd.DataFrame, n_clusters: int = 100,
                       geohash_precision: int = 6) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Two-stage clustering: geohash aggregation then K-means.
        Efficient for very large datasets.
        """
        # Stage 1: Aggregate by geohash
        df = df.copy()
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], geohash_precision), axis=1)

        # Get geohash centroids
        geohash_stats = df.groupby('geohash').agg({
            'lat': 'mean',
            'lon': 'mean',
            'Speed': 'mean'
        }).reset_index()
        geohash_stats['count'] = df.groupby('geohash').size().values

        # Stage 2: K-means on geohash centroids
        coords = geohash_stats[['lat', 'lon']].values
        kmeans = MiniBatchKMeans(n_clusters=min(n_clusters, len(coords)),
                                  random_state=42, batch_size=1000)
        geohash_stats['cluster'] = kmeans.fit_predict(coords)

        # Map back to original points
        geohash_to_cluster = dict(zip(geohash_stats['geohash'], geohash_stats['cluster']))
        labels = df['geohash'].map(geohash_to_cluster).values

        return labels, geohash_stats


class HotspotAnalysis:
    """Identify spatial hotspots and cold spots."""

    @staticmethod
    def kernel_density_grid(df: pd.DataFrame, grid_size: int = 100,
                            bandwidth: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute kernel density estimate on a grid.

        Returns:
            xx, yy: Grid coordinates
            density: Density values
        """
        from scipy.stats import gaussian_kde

        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        lon_min, lon_max = df['lon'].min(), df['lon'].max()

        # Create grid
        xx, yy = np.mgrid[lon_min:lon_max:complex(grid_size),
                          lat_min:lat_max:complex(grid_size)]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        # Sample if too large
        sample_size = min(100000, len(df))
        sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

        values = np.vstack([sample['lon'].values, sample['lat'].values])
        kernel = gaussian_kde(values, bw_method=bandwidth)
        density = np.reshape(kernel(positions).T, xx.shape)

        return xx, yy, density

    @staticmethod
    def getis_ord_gi(df: pd.DataFrame, geohash_precision: int = 6) -> pd.DataFrame:
        """
        Getis-Ord Gi* statistic for hotspot detection.
        Computed on geohash-aggregated data for efficiency.
        """
        df = df.copy()
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], geohash_precision), axis=1)

        # Aggregate counts
        counts = df.groupby('geohash').size().reset_index(name='count')
        counts['lat'] = df.groupby('geohash')['lat'].mean().values
        counts['lon'] = df.groupby('geohash')['lon'].mean().values

        n = len(counts)
        x = counts['count'].values
        x_mean = x.mean()
        s = x.std()

        # Build neighbor weights (queen contiguity approximation using distance)
        coords = np.radians(counts[['lat', 'lon']].values)
        tree = BallTree(coords, metric='haversine')

        # ~1km neighbor threshold
        threshold_rad = 1.0 / EARTH_RADIUS_KM
        neighbors = tree.query_radius(coords, r=threshold_rad)

        gi_scores = []
        for i in range(n):
            neighbor_idx = neighbors[i]
            w_sum = len(neighbor_idx)
            if w_sum == 0:
                gi_scores.append(0)
                continue

            wx_sum = x[neighbor_idx].sum()
            numerator = wx_sum - x_mean * w_sum

            s_star = np.sqrt((n * w_sum - w_sum**2) / (n - 1))
            denominator = s * s_star if s_star > 0 else 1

            gi = numerator / denominator if denominator > 0 else 0
            gi_scores.append(gi)

        counts['gi_score'] = gi_scores
        counts['hotspot'] = pd.cut(counts['gi_score'],
                                    bins=[-np.inf, -2.58, -1.96, 1.96, 2.58, np.inf],
                                    labels=['Cold99', 'Cold95', 'NotSig', 'Hot95', 'Hot99'])
        return counts


class MovementPatterns:
    """Analyze movement patterns and trajectories."""

    @staticmethod
    def detect_stay_points(df: pd.DataFrame, time_threshold_min: float = 10,
                          distance_threshold_km: float = 0.05) -> pd.DataFrame:
        """
        Detect stay points (locations where subject stayed for extended time).

        Args:
            df: DataFrame with 'lat', 'lon', 'Time' columns
            time_threshold_min: Minimum time to consider a stay (minutes)
            distance_threshold_km: Maximum roaming distance for stay
        """
        df = df.sort_values('Time').copy()
        df['Time'] = pd.to_datetime(df['Time'])

        stay_points = []
        i = 0
        n = len(df)

        while i < n:
            j = i + 1
            while j < n:
                dist = haversine_distance(
                    df.iloc[i]['lat'], df.iloc[i]['lon'],
                    df.iloc[j]['lat'], df.iloc[j]['lon']
                )
                if dist > distance_threshold_km:
                    break
                j += 1

            if j > i + 1:
                time_diff = (df.iloc[j-1]['Time'] - df.iloc[i]['Time']).total_seconds() / 60
                if time_diff >= time_threshold_min:
                    stay_points.append({
                        'lat': df.iloc[i:j]['lat'].mean(),
                        'lon': df.iloc[i:j]['lon'].mean(),
                        'start_time': df.iloc[i]['Time'],
                        'end_time': df.iloc[j-1]['Time'],
                        'duration_min': time_diff,
                        'n_points': j - i
                    })
            i = j

        return pd.DataFrame(stay_points)

    @staticmethod
    def compute_speed_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Compute speed statistics by various groupings."""
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'])
        df['hour'] = df['Time'].dt.hour
        df['day_of_week'] = df['Time'].dt.dayofweek
        df['month'] = df['Time'].dt.month

        stats_list = []

        # By hour
        hourly = df.groupby('hour')['Speed'].agg(['mean', 'std', 'median', 'count'])
        hourly['group_type'] = 'hour'
        hourly['group_value'] = hourly.index
        stats_list.append(hourly)

        # By day of week
        daily = df.groupby('day_of_week')['Speed'].agg(['mean', 'std', 'median', 'count'])
        daily['group_type'] = 'day_of_week'
        daily['group_value'] = daily.index
        stats_list.append(daily)

        # By activity
        if 'Activity' in df.columns:
            activity = df.groupby('Activity')['Speed'].agg(['mean', 'std', 'median', 'count'])
            activity['group_type'] = 'activity'
            activity['group_value'] = activity.index
            stats_list.append(activity)

        return pd.concat(stats_list, ignore_index=True)

    @staticmethod
    def trajectory_segmentation(df: pd.DataFrame, gap_threshold_min: float = 30) -> np.ndarray:
        """
        Segment trajectory into trips based on time gaps.

        Returns array of segment IDs for each point.
        """
        df = df.sort_values('Time').copy()
        df['Time'] = pd.to_datetime(df['Time'])

        time_diffs = df['Time'].diff().dt.total_seconds() / 60
        segment_starts = (time_diffs > gap_threshold_min) | (time_diffs.isna())
        segments = segment_starts.cumsum()

        return segments.values


class HomeRangeEstimation:
    """Estimate home range using various methods."""

    @staticmethod
    def minimum_convex_polygon(df: pd.DataFrame, percentile: float = 95) -> dict:
        """
        Compute Minimum Convex Polygon (MCP) for home range.

        Args:
            df: DataFrame with 'lat' and 'lon'
            percentile: Percentile of points to include (e.g., 95%)
        """
        # Find centroid
        centroid_lat = df['lat'].mean()
        centroid_lon = df['lon'].mean()

        # Compute distances from centroid
        distances = haversine_vectorized(
            df['lat'].values, df['lon'].values,
            np.full(len(df), centroid_lat),
            np.full(len(df), centroid_lon)
        )

        # Filter to percentile
        threshold = np.percentile(distances, percentile)
        mask = distances <= threshold
        filtered = df[mask]

        if len(filtered) < 3:
            return {'area_km2': 0, 'vertices': [], 'centroid': (centroid_lat, centroid_lon)}

        # Compute convex hull
        points = filtered[['lon', 'lat']].values
        try:
            hull = ConvexHull(points)

            # Approximate area (using simple polygon area formula)
            # Convert to approximate km using local scale
            lat_scale = 111.0  # km per degree latitude
            lon_scale = 111.0 * np.cos(np.radians(centroid_lat))

            scaled_points = points.copy()
            scaled_points[:, 0] *= lon_scale
            scaled_points[:, 1] *= lat_scale
            scaled_hull = ConvexHull(scaled_points)

            vertices = points[hull.vertices].tolist()

            return {
                'area_km2': scaled_hull.volume,  # In 2D, volume is area
                'vertices': vertices,
                'centroid': (centroid_lat, centroid_lon),
                'n_points': len(filtered)
            }
        except Exception as e:
            return {'area_km2': 0, 'vertices': [], 'centroid': (centroid_lat, centroid_lon), 'error': str(e)}

    @staticmethod
    def utilization_distribution(df: pd.DataFrame, grid_size: int = 50) -> dict:
        """
        Compute utilization distribution (UD) using kernel density.
        Returns contour levels for different percentages of UD.
        """
        from scipy.stats import gaussian_kde

        sample_size = min(50000, len(df))
        sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

        values = np.vstack([sample['lon'].values, sample['lat'].values])
        kernel = gaussian_kde(values)

        lat_min, lat_max = df['lat'].min(), df['lat'].max()
        lon_min, lon_max = df['lon'].min(), df['lon'].max()

        xx, yy = np.mgrid[lon_min:lon_max:complex(grid_size),
                          lat_min:lat_max:complex(grid_size)]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kernel(positions).T, xx.shape)

        # Normalize to sum to 1
        density = density / density.sum()

        # Find contour levels for different UD percentages
        sorted_density = np.sort(density.ravel())[::-1]
        cumsum = np.cumsum(sorted_density)

        levels = {}
        for pct in [50, 75, 90, 95]:
            idx = np.searchsorted(cumsum, pct / 100)
            if idx < len(sorted_density):
                levels[pct] = sorted_density[idx]

        return {
            'grid_x': xx,
            'grid_y': yy,
            'density': density,
            'contour_levels': levels
        }


class TemporalPatterns:
    """Analyze temporal patterns in movement data."""

    @staticmethod
    def hourly_activity_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create hour x day-of-week activity matrix.
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'])
        df['hour'] = df['Time'].dt.hour
        df['day_of_week'] = df['Time'].dt.dayofweek

        matrix = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        # Normalize by row (day)
        matrix = matrix.div(matrix.sum(axis=1), axis=0)

        return matrix

    @staticmethod
    def activity_rhythm(df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze activity rhythm patterns.
        Returns hourly average speed and activity counts.
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'])
        df['hour'] = df['Time'].dt.hour

        rhythm = df.groupby('hour').agg({
            'Speed': ['mean', 'std', 'median'],
            'lat': 'count'
        })
        rhythm.columns = ['speed_mean', 'speed_std', 'speed_median', 'count']
        rhythm['activity_index'] = rhythm['speed_mean'] / rhythm['speed_mean'].max()

        return rhythm

    @staticmethod
    def seasonal_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Analyze seasonal movement patterns."""
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'])
        df['month'] = df['Time'].dt.month
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        seasonal = df.groupby('season').agg({
            'Speed': ['mean', 'median'],
            'lat': ['std', 'count'],
            'lon': 'std'
        })
        seasonal.columns = ['speed_mean', 'speed_median', 'lat_spread', 'count', 'lon_spread']

        return seasonal


class GridAggregation:
    """Grid-based spatial aggregation for large datasets."""

    @staticmethod
    def geohash_aggregate(df: pd.DataFrame, precision: int = 6,
                          agg_cols: dict = None) -> pd.DataFrame:
        """
        Aggregate data by geohash cells.

        Args:
            df: Input DataFrame
            precision: Geohash precision (5=~5km, 6=~1km, 7=~150m)
            agg_cols: Dict of column: aggregation function pairs
        """
        df = df.copy()
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], precision), axis=1)

        default_agg = {
            'lat': 'mean',
            'lon': 'mean',
            'Speed': ['mean', 'std', 'max'],
        }
        agg_cols = agg_cols or default_agg

        aggregated = df.groupby('geohash').agg(agg_cols)
        aggregated.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                              for col in aggregated.columns]
        aggregated['count'] = df.groupby('geohash').size()
        aggregated = aggregated.reset_index()

        return aggregated

    @staticmethod
    def hex_grid_aggregate(df: pd.DataFrame, resolution: float = 0.01) -> pd.DataFrame:
        """
        Simple hexagonal grid aggregation.
        Uses offset coordinates to create hex-like bins.
        """
        df = df.copy()

        # Create hex grid indices
        df['hex_x'] = (df['lon'] / resolution).astype(int)
        df['hex_y'] = (df['lat'] / (resolution * np.sqrt(3) / 2)).astype(int)
        # Offset every other row
        df['hex_x'] = df['hex_x'] + (df['hex_y'] % 2) * 0.5

        df['hex_id'] = df['hex_x'].astype(str) + '_' + df['hex_y'].astype(str)

        aggregated = df.groupby('hex_id').agg({
            'lat': 'mean',
            'lon': 'mean',
            'Speed': ['mean', 'count']
        })
        aggregated.columns = ['lat', 'lon', 'speed_mean', 'count']
        aggregated = aggregated.reset_index()

        return aggregated


class SpatialStatistics:
    """Spatial statistical measures."""

    @staticmethod
    def global_morans_i(df: pd.DataFrame, value_col: str = 'Speed',
                        geohash_precision: int = 6, k_neighbors: int = 8) -> dict:
        """
        Compute Global Moran's I for spatial autocorrelation.
        Uses geohash aggregation for efficiency.
        """
        # Aggregate by geohash
        df = df.copy()
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], geohash_precision), axis=1)

        agg = df.groupby('geohash').agg({
            'lat': 'mean',
            'lon': 'mean',
            value_col: 'mean'
        }).reset_index()

        if len(agg) < k_neighbors + 1:
            return {'morans_i': np.nan, 'p_value': np.nan, 'z_score': np.nan}

        # Build spatial weights using k-nearest neighbors
        coords = np.radians(agg[['lat', 'lon']].values)
        tree = BallTree(coords, metric='haversine')

        distances, indices = tree.query(coords, k=k_neighbors + 1)

        n = len(agg)
        x = agg[value_col].values
        x_mean = x.mean()
        x_dev = x - x_mean

        # Compute Moran's I
        numerator = 0
        w_sum = 0

        for i in range(n):
            for j in indices[i, 1:]:  # Skip self
                w = 1  # Binary weights
                numerator += w * x_dev[i] * x_dev[j]
                w_sum += w

        denominator = (x_dev ** 2).sum()
        morans_i = (n / w_sum) * (numerator / denominator) if denominator > 0 else 0

        # Expected value and variance for significance testing
        e_i = -1 / (n - 1)

        # Simplified variance calculation
        s1 = 2 * w_sum
        s2 = 4 * k_neighbors * n

        var_i = (n * s1 - s2) / ((n - 1) * (n + 1) * w_sum**2)
        z_score = (morans_i - e_i) / np.sqrt(var_i) if var_i > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return {
            'morans_i': morans_i,
            'expected_i': e_i,
            'z_score': z_score,
            'p_value': p_value,
            'interpretation': 'Clustered' if morans_i > e_i else 'Dispersed'
        }


def load_data(filepath: str, sample_frac: float = 1.0,
              nrows: Optional[int] = None) -> pd.DataFrame:
    """Load data efficiently from gzipped TSV."""
    print(f"Loading data from {filepath}...")

    if filepath.endswith('.gz'):
        df = pd.read_csv(filepath, sep='\t', compression='gzip', nrows=nrows)
    else:
        df = pd.read_csv(filepath, sep='\t', nrows=nrows)

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)

    print(f"Loaded {len(df):,} records")
    return df


def run_analysis(df: pd.DataFrame, output_dir: str, name_filter: Optional[str] = None):
    """Run comprehensive geospatial analysis suite."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Filter by name if specified
    if name_filter and 'Name' in df.columns:
        df = df[df['Name'] == name_filter].copy()
        print(f"Filtered to {len(df):,} records for {name_filter}")

    results = {}

    # 1. Geohash clustering and aggregation
    print("\n1. Running geohash-based K-means clustering...")
    labels, geohash_stats = SpatialClustering.geohash_kmeans(df, n_clusters=50)
    df['cluster'] = labels
    geohash_stats.to_csv(output_path / 'geohash_clusters.csv', index=False)
    results['n_clusters'] = len(geohash_stats['cluster'].unique())
    print(f"   Found {results['n_clusters']} clusters")

    # 2. Hotspot analysis
    print("\n2. Running hotspot analysis (Getis-Ord Gi*)...")
    hotspots = HotspotAnalysis.getis_ord_gi(df, geohash_precision=6)
    hotspots.to_csv(output_path / 'hotspots.csv', index=False)
    results['n_hotspots'] = len(hotspots[hotspots['hotspot'].isin(['Hot95', 'Hot99'])])
    results['n_coldspots'] = len(hotspots[hotspots['hotspot'].isin(['Cold95', 'Cold99'])])
    print(f"   Found {results['n_hotspots']} significant hotspots, {results['n_coldspots']} coldspots")

    # 3. Movement pattern statistics
    print("\n3. Computing movement statistics...")
    speed_stats = MovementPatterns.compute_speed_stats(df)
    speed_stats.to_csv(output_path / 'speed_stats.csv', index=False)

    # 4. Trajectory segmentation
    if 'Time' in df.columns:
        print("\n4. Segmenting trajectories...")
        segments = MovementPatterns.trajectory_segmentation(df)
        df['segment'] = segments
        results['n_segments'] = len(np.unique(segments))
        print(f"   Found {results['n_segments']} trajectory segments")

    # 5. Home range estimation
    print("\n5. Estimating home range (MCP)...")
    mcp = HomeRangeEstimation.minimum_convex_polygon(df, percentile=95)
    results['home_range_km2'] = mcp['area_km2']
    print(f"   95% MCP area: {mcp['area_km2']:.2f} km²")

    # 6. Temporal patterns
    if 'Time' in df.columns:
        print("\n6. Analyzing temporal patterns...")
        hourly_matrix = TemporalPatterns.hourly_activity_matrix(df)
        hourly_matrix.to_csv(output_path / 'hourly_activity_matrix.csv')

        rhythm = TemporalPatterns.activity_rhythm(df)
        rhythm.to_csv(output_path / 'activity_rhythm.csv')

        seasonal = TemporalPatterns.seasonal_patterns(df)
        seasonal.to_csv(output_path / 'seasonal_patterns.csv')

    # 7. Grid aggregation
    print("\n7. Creating grid aggregations...")
    geohash_agg = GridAggregation.geohash_aggregate(df, precision=6)
    geohash_agg.to_csv(output_path / 'geohash_aggregation.csv', index=False)
    results['n_geohash_cells'] = len(geohash_agg)
    print(f"   Created {results['n_geohash_cells']} geohash cells")

    # 8. Spatial autocorrelation
    print("\n8. Computing spatial autocorrelation (Moran's I)...")
    morans = SpatialStatistics.global_morans_i(df, value_col='Speed')
    results['morans_i'] = morans['morans_i']
    results['morans_interpretation'] = morans['interpretation']
    print(f"   Moran's I: {morans['morans_i']:.4f} ({morans['interpretation']})")

    # Save summary results
    with open(output_path / 'analysis_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save clustered data
    print("\n9. Saving enriched dataset...")
    df.to_csv(output_path / 'enriched_data.csv.gz', compression='gzip', index=False)

    print(f"\n✓ Analysis complete! Results saved to {output_path}")
    return results, df


def main():
    parser = argparse.ArgumentParser(description='Geospatial Analysis for CatDash')
    parser.add_argument('-i', '--input', type=str, default='output/raw.tsv.gz',
                        help='Input file (gzipped TSV)')
    parser.add_argument('-o', '--output', type=str, default='output/analysis',
                        help='Output directory')
    parser.add_argument('-s', '--sample', type=float, default=1.0,
                        help='Sample fraction (0-1) for large datasets')
    parser.add_argument('-n', '--nrows', type=int, default=None,
                        help='Number of rows to read (for testing)')
    parser.add_argument('--name', type=str, default=None,
                        help='Filter by Name column')
    parser.add_argument('--analysis', type=str, default='all',
                        choices=['all', 'cluster', 'hotspot', 'movement', 'homerange',
                                 'temporal', 'grid', 'spatial'],
                        help='Which analysis to run')

    args = parser.parse_args()

    # Load data
    df = load_data(args.input, sample_frac=args.sample, nrows=args.nrows)

    # Run analysis
    results, enriched_df = run_analysis(df, args.output, name_filter=args.name)

    return results


if __name__ == '__main__':
    main()
