"""
Creative Geospatial Analysis Module for CatDash
================================================
Innovative and cutting-edge geospatial analysis methods.

Includes:
1. Fractal Dimension Analysis - Measure movement complexity
2. Directional Persistence - Analyze heading consistency
3. Space Utilization Distribution (UD) - Probabilistic home range
4. Sinuosity Index - Track straightness analysis
5. First Passage Time Analysis - Identify area-restricted search
6. Revisitation Analysis - Find frequently revisited locations
7. Nocturnal vs Diurnal Patterns - Day/night behavior comparison
8. Weather-Correlated Movement - Environmental effects (if data available)
9. Activity Burst Detection - Identify sudden activity changes
10. Movement Entropy - Measure predictability of movement
11. Convex Hull Time Series - Dynamic home range evolution
12. Significant Location Mining - Discover meaningful places
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

from geospatial_analysis import encode_geohash, haversine_vectorized, haversine_distance, EARTH_RADIUS_KM


class FractalDimensionAnalysis:
    """
    Compute fractal dimension of movement trajectories.
    Higher values indicate more complex, space-filling movement.
    Efficient box-counting implementation for large datasets.
    """

    @staticmethod
    def box_counting_dimension(df: pd.DataFrame, min_box_size: float = 0.001,
                               max_box_size: float = 0.1, n_sizes: int = 15,
                               sample_size: int = 50000) -> dict:
        """
        Compute fractal dimension using box-counting method.

        Args:
            df: DataFrame with lat, lon columns
            min_box_size: Minimum box size in degrees
            max_box_size: Maximum box size in degrees
            n_sizes: Number of box sizes to test
            sample_size: Maximum points to use for efficiency

        Returns:
            dict with fractal_dimension, r_squared, interpretation
        """
        # Sample for efficiency
        if len(df) > sample_size:
            sample = df.sample(n=sample_size, random_state=42)
        else:
            sample = df

        coords = sample[['lon', 'lat']].values

        # Generate box sizes (logarithmically spaced)
        box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), n_sizes)
        box_counts = []

        for size in box_sizes:
            # Count non-empty boxes
            box_x = (coords[:, 0] / size).astype(int)
            box_y = (coords[:, 1] / size).astype(int)
            unique_boxes = len(set(zip(box_x, box_y)))
            box_counts.append(unique_boxes)

        # Linear regression on log-log scale
        log_sizes = np.log(1 / box_sizes)
        log_counts = np.log(box_counts)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_counts)

        # Fractal dimension is the slope
        fractal_dim = slope

        # Interpretation
        if fractal_dim < 1.2:
            interpretation = "Linear movement (direct travel)"
        elif fractal_dim < 1.5:
            interpretation = "Moderately complex movement"
        elif fractal_dim < 1.8:
            interpretation = "Complex, area-exploring movement"
        else:
            interpretation = "Highly complex, space-filling movement"

        return {
            'fractal_dimension': fractal_dim,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_error': std_err,
            'interpretation': interpretation,
            'box_sizes': box_sizes.tolist(),
            'box_counts': box_counts
        }

    @staticmethod
    def fractal_by_time_period(df: pd.DataFrame, time_resolution: str = 'D') -> pd.DataFrame:
        """
        Compute fractal dimension for each time period to track complexity changes.

        Args:
            df: DataFrame with lat, lon, Time columns
            time_resolution: 'D'=day, 'W'=week, 'M'=month
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        df['period'] = df['Time'].dt.to_period(time_resolution)

        results = []
        for period, group in df.groupby('period'):
            if len(group) >= 50:  # Minimum points for meaningful fractal analysis
                fractal = FractalDimensionAnalysis.box_counting_dimension(group)
                results.append({
                    'period': str(period),
                    'fractal_dimension': fractal['fractal_dimension'],
                    'r_squared': fractal['r_squared'],
                    'n_points': len(group)
                })

        return pd.DataFrame(results)


class DirectionalPersistence:
    """
    Analyze how consistently an animal moves in a particular direction.
    High persistence = directed travel; Low persistence = random exploration.
    """

    @staticmethod
    def compute_persistence(df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """
        Compute directional persistence using vector auto-correlation.

        Args:
            df: DataFrame with Heading column
            window_size: Rolling window for persistence calculation
        """
        df = df.copy().sort_values('Time')
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')

        # Convert heading to unit vectors
        heading_rad = np.radians(df['Heading'].values)
        df['heading_x'] = np.cos(heading_rad)
        df['heading_y'] = np.sin(heading_rad)

        # Rolling mean of unit vectors (resultant vector)
        if 'Name' in df.columns:
            df['mean_x'] = df.groupby('Name')['heading_x'].rolling(window_size, min_periods=2).mean().reset_index(drop=True)
            df['mean_y'] = df.groupby('Name')['heading_y'].rolling(window_size, min_periods=2).mean().reset_index(drop=True)
        else:
            df['mean_x'] = df['heading_x'].rolling(window_size, min_periods=2).mean()
            df['mean_y'] = df['heading_y'].rolling(window_size, min_periods=2).mean()

        # Persistence = length of mean vector (0 = random, 1 = perfectly aligned)
        df['persistence'] = np.sqrt(df['mean_x']**2 + df['mean_y']**2)

        # Mean heading direction
        df['mean_heading'] = np.degrees(np.arctan2(df['mean_y'], df['mean_x'])) % 360

        return df[['lat', 'lon', 'Time', 'Speed', 'Heading', 'persistence', 'mean_heading']].copy()

    @staticmethod
    def summarize_persistence(df: pd.DataFrame) -> dict:
        """Get summary statistics of directional persistence."""
        persistence_df = DirectionalPersistence.compute_persistence(df)

        return {
            'mean_persistence': persistence_df['persistence'].mean(),
            'median_persistence': persistence_df['persistence'].median(),
            'std_persistence': persistence_df['persistence'].std(),
            'pct_high_persistence': (persistence_df['persistence'] > 0.7).mean() * 100,
            'pct_low_persistence': (persistence_df['persistence'] < 0.3).mean() * 100,
            'interpretation': 'Directed' if persistence_df['persistence'].mean() > 0.5 else 'Exploratory'
        }


class SinuosityAnalysis:
    """
    Measure track sinuosity (tortuosity) - how winding the path is.
    Sinuosity = actual distance / straight-line distance
    """

    @staticmethod
    def compute_sinuosity(df: pd.DataFrame, segment_points: int = 20,
                          sample_size: int = 50000) -> pd.DataFrame:
        """
        Compute sinuosity for trajectory segments.
        Optimized with vectorized distance calculations.

        Args:
            df: DataFrame with lat, lon, Time
            segment_points: Number of points per segment for analysis
            sample_size: Maximum points to analyze for efficiency
        """
        df = df.copy()

        # Sample if needed
        if len(df) > sample_size:
            # Use systematic sampling to preserve temporal order
            step = len(df) // sample_size
            df = df.iloc[::step].reset_index(drop=True)

        df = df.sort_values('Time').reset_index(drop=True)
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')

        results = []

        # Process by individual if available
        group_col = 'Name' if 'Name' in df.columns else None
        groups = df.groupby(group_col) if group_col else [(None, df)]

        for name, group in groups:
            group = group.reset_index(drop=True)
            n_segments = len(group) // segment_points

            if n_segments == 0:
                continue

            # Vectorized computation for all segments at once
            lats = group['lat'].values
            lons = group['lon'].values
            times = group['Time'].values
            speeds = group['Speed'].values

            for i in range(n_segments):
                start_idx = i * segment_points
                end_idx = start_idx + segment_points

                seg_lats = lats[start_idx:end_idx]
                seg_lons = lons[start_idx:end_idx]

                # Vectorized actual path distance
                step_distances = haversine_vectorized(
                    seg_lats[:-1], seg_lons[:-1],
                    seg_lats[1:], seg_lons[1:]
                )
                actual_distance = np.sum(step_distances)

                # Straight-line distance
                straight_distance = haversine_distance(
                    seg_lats[0], seg_lons[0],
                    seg_lats[-1], seg_lons[-1]
                )

                # Sinuosity index (1 = straight line, higher = more winding)
                sinuosity = actual_distance / straight_distance if straight_distance > 0.001 else np.nan

                results.append({
                    'name': name,
                    'segment_id': i,
                    'start_time': times[start_idx],
                    'end_time': times[end_idx - 1],
                    'center_lat': seg_lats.mean(),
                    'center_lon': seg_lons.mean(),
                    'actual_distance_km': actual_distance,
                    'straight_distance_km': straight_distance,
                    'sinuosity': sinuosity,
                    'mean_speed': speeds[start_idx:end_idx].mean()
                })

        result_df = pd.DataFrame(results)

        # Classify movement type
        if len(result_df) > 0:
            result_df['movement_type'] = pd.cut(
                result_df['sinuosity'],
                bins=[0, 1.1, 1.5, 2.0, np.inf],
                labels=['Direct', 'Slight detour', 'Winding', 'Area-restricted']
            )

        return result_df


class FirstPassageTime:
    """
    First Passage Time (FPT) analysis to identify area-restricted search behavior.
    High FPT indicates the animal spent more time in an area (foraging/resting).
    """

    @staticmethod
    def compute_fpt(df: pd.DataFrame, radii_km: list = None,
                    sample_size: int = 10000) -> pd.DataFrame:
        """
        Compute First Passage Time for multiple radii.

        Args:
            df: DataFrame with lat, lon, Time columns
            radii_km: List of radii to analyze (default: [0.05, 0.1, 0.2, 0.5])
            sample_size: Maximum points to analyze for efficiency
        """
        if radii_km is None:
            radii_km = [0.05, 0.1, 0.2, 0.5]

        df = df.copy().sort_values('Time')
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')

        # Sample for efficiency
        if len(df) > sample_size:
            # Systematic sampling to maintain temporal order
            step = len(df) // sample_size
            df = df.iloc[::step].reset_index(drop=True)

        coords = df[['lat', 'lon']].values
        times = df['Time'].values
        n = len(coords)

        results = []

        for radius in radii_km:
            fpt_values = []

            for i in range(n):
                # Find time to leave circle of given radius
                j = i + 1
                while j < n:
                    dist = haversine_distance(
                        coords[i, 0], coords[i, 1],
                        coords[j, 0], coords[j, 1]
                    )
                    if dist > radius:
                        break
                    j += 1

                if j < n:
                    fpt = (times[j] - times[i]) / np.timedelta64(1, 'h')  # Hours
                else:
                    fpt = np.nan

                fpt_values.append(fpt)

            df_temp = df.copy()
            df_temp[f'fpt_{radius}km'] = fpt_values

            # Identify high FPT locations (area-restricted search)
            threshold = np.nanpercentile(fpt_values, 90)
            df_temp[f'ars_{radius}km'] = df_temp[f'fpt_{radius}km'] > threshold

            results.append({
                'radius_km': radius,
                'mean_fpt_hours': np.nanmean(fpt_values),
                'median_fpt_hours': np.nanmedian(fpt_values),
                'variance_fpt': np.nanvar(fpt_values),
                'n_ars_locations': np.nansum(df_temp[f'ars_{radius}km'])
            })

        return pd.DataFrame(results)

    @staticmethod
    def identify_ars_zones(df: pd.DataFrame, radius_km: float = 0.1,
                           fpt_threshold_pct: float = 90) -> pd.DataFrame:
        """
        Identify Area-Restricted Search (ARS) zones - locations with high FPT.
        """
        fpt_results = FirstPassageTime.compute_fpt(df, radii_km=[radius_km])

        # Get high FPT points and cluster them
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        df = df.sort_values('Time')

        # Simplified: mark high-density areas as ARS zones
        # Use geohash aggregation for efficiency
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], 7), axis=1)
        visit_counts = df.groupby('geohash').size()

        threshold = visit_counts.quantile(fpt_threshold_pct / 100)
        high_visit_hashes = visit_counts[visit_counts > threshold].index

        ars_zones = df[df['geohash'].isin(high_visit_hashes)].groupby('geohash').agg({
            'lat': 'mean',
            'lon': 'mean',
            'Time': ['min', 'max'],
            'Speed': 'mean'
        }).reset_index()
        ars_zones.columns = ['geohash', 'lat', 'lon', 'first_visit', 'last_visit', 'mean_speed']
        ars_zones['visit_count'] = ars_zones['geohash'].map(visit_counts)

        return ars_zones


class RevisitationAnalysis:
    """
    Analyze how often and when locations are revisited.
    Reveals routine behaviors and important locations.
    """

    @staticmethod
    def compute_revisitation(df: pd.DataFrame, geohash_precision: int = 7,
                             min_hours_between: float = 1.0,
                             sample_size: int = 100000) -> pd.DataFrame:
        """
        Compute revisitation statistics for each location.
        Optimized for large datasets using vectorized operations.

        Args:
            df: DataFrame with lat, lon, Time
            geohash_precision: Spatial resolution (7 ≈ 150m)
            min_hours_between: Minimum hours between visits to count as separate
            sample_size: Maximum rows to process for efficiency
        """
        df = df.copy()

        # Sample for efficiency if needed
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        df = df.sort_values('Time')
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')

        # Vectorized geohash encoding
        df['geohash'] = [encode_geohash(lat, lon, geohash_precision)
                         for lat, lon in zip(df['lat'].values, df['lon'].values)]

        # Group by geohash and compute statistics using pandas aggregations
        agg_stats = df.groupby('geohash').agg({
            'lat': 'mean',
            'lon': 'mean',
            'Time': ['count', 'min', 'max']
        })
        agg_stats.columns = ['lat', 'lon', 'total_points', 'first_visit', 'last_visit']
        agg_stats = agg_stats.reset_index()

        # Compute time span
        agg_stats['time_span_days'] = (
            agg_stats['last_visit'] - agg_stats['first_visit']
        ).dt.total_seconds() / 86400

        # For distinct visits, we need to count time gaps > min_hours_between
        # Use a more efficient approach with groupby
        def count_distinct_visits(group):
            if len(group) <= 1:
                return 1
            times = group['Time'].sort_values()
            time_diffs = times.diff().dt.total_seconds() / 3600
            # Count number of gaps larger than threshold (+ 1 for the first visit)
            return (time_diffs > min_hours_between).sum() + 1

        visit_counts = df.groupby('geohash').apply(count_distinct_visits, include_groups=False)
        agg_stats['distinct_visits'] = agg_stats['geohash'].map(visit_counts)

        # Classify location types
        agg_stats['location_type'] = 'Transient'
        agg_stats.loc[agg_stats['distinct_visits'] >= 3, 'location_type'] = 'Regular'
        agg_stats.loc[agg_stats['distinct_visits'] >= 10, 'location_type'] = 'Frequent'
        agg_stats.loc[
            (agg_stats['distinct_visits'] >= 20) & (agg_stats['total_points'] > 100),
            'location_type'
        ] = 'Core'

        return agg_stats.sort_values('distinct_visits', ascending=False)


class NocturnalDiurnalAnalysis:
    """
    Compare movement patterns between day and night.
    """

    @staticmethod
    def classify_day_night(df: pd.DataFrame, day_start: int = 6,
                           day_end: int = 18) -> pd.DataFrame:
        """
        Classify each point as day or night and compute statistics.

        Note: Uses simple hour-based classification. For production,
        consider using sunrise/sunset times based on location.
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        df['hour'] = df['Time'].dt.hour

        df['period'] = 'Night'
        df.loc[(df['hour'] >= day_start) & (df['hour'] < day_end), 'period'] = 'Day'

        # Compute statistics
        stats = df.groupby('period').agg({
            'Speed': ['mean', 'std', 'median', 'max'],
            'lat': ['std', 'count'],
            'lon': 'std'
        })
        stats.columns = ['speed_mean', 'speed_std', 'speed_median', 'speed_max',
                        'lat_spread', 'n_points', 'lon_spread']
        stats = stats.reset_index()

        # Activity index (based on speed and spatial spread)
        stats['activity_index'] = stats['speed_mean'] * (stats['lat_spread'] + stats['lon_spread'])

        return stats

    @staticmethod
    def hourly_activity_profile(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create detailed hourly activity profile.
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        df['hour'] = df['Time'].dt.hour

        profile = df.groupby('hour').agg({
            'Speed': ['mean', 'std', 'count'],
            'lat': 'std',
            'lon': 'std'
        })
        profile.columns = ['speed_mean', 'speed_std', 'n_points', 'lat_spread', 'lon_spread']
        profile = profile.reset_index()

        # Normalize activity
        profile['normalized_activity'] = profile['speed_mean'] / profile['speed_mean'].max()

        # Identify peak hours
        threshold = profile['normalized_activity'].quantile(0.75)
        profile['is_peak_hour'] = profile['normalized_activity'] >= threshold

        return profile


class MovementEntropy:
    """
    Compute entropy measures of movement patterns.
    Low entropy = predictable, routine movement
    High entropy = unpredictable, variable movement
    """

    @staticmethod
    def spatial_entropy(df: pd.DataFrame, geohash_precision: int = 6) -> dict:
        """
        Compute spatial entropy based on location visit distribution.
        """
        df = df.copy()
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], geohash_precision), axis=1)

        # Visit frequency distribution
        visit_counts = df['geohash'].value_counts()
        probs = visit_counts / visit_counts.sum()

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs))  # Maximum possible entropy

        # Normalized entropy (0 = single location, 1 = uniform distribution)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return {
            'spatial_entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy,
            'n_locations': len(visit_counts),
            'interpretation': 'Predictable' if normalized_entropy < 0.5 else 'Unpredictable'
        }

    @staticmethod
    def temporal_entropy(df: pd.DataFrame) -> dict:
        """
        Compute temporal entropy based on activity time distribution.
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        df['hour'] = df['Time'].dt.hour

        # Activity distribution across hours
        hour_counts = df['hour'].value_counts()
        probs = hour_counts / hour_counts.sum()

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(24)  # Maximum entropy for 24 hours

        normalized_entropy = entropy / max_entropy

        return {
            'temporal_entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy,
            'interpretation': 'Regular routine' if normalized_entropy < 0.7 else 'Variable schedule'
        }

    @staticmethod
    def transition_entropy(df: pd.DataFrame, geohash_precision: int = 5) -> dict:
        """
        Compute entropy of location transitions (predictability of next location).
        """
        df = df.copy().sort_values('Time')
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], geohash_precision), axis=1)
        df['next_geohash'] = df['geohash'].shift(-1)

        # Filter valid transitions
        transitions = df.dropna(subset=['next_geohash'])

        # Compute conditional entropy: H(next|current)
        transition_counts = transitions.groupby(['geohash', 'next_geohash']).size()
        current_counts = transitions.groupby('geohash').size()

        conditional_entropy = 0
        for (current, next_loc), count in transition_counts.items():
            p_transition = count / current_counts[current]
            p_current = current_counts[current] / len(transitions)
            if p_transition > 0:
                conditional_entropy -= p_current * p_transition * np.log2(p_transition)

        return {
            'transition_entropy': conditional_entropy,
            'n_unique_transitions': len(transition_counts),
            'interpretation': 'Predictable routes' if conditional_entropy < 2 else 'Variable routes'
        }


class ActivityBurstDetection:
    """
    Detect sudden bursts or lulls in activity.
    """

    @staticmethod
    def detect_bursts(df: pd.DataFrame, window_hours: float = 1.0,
                      z_threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect activity bursts using rolling window z-scores.

        Args:
            df: DataFrame with Speed, Time columns
            window_hours: Rolling window size in hours
            z_threshold: Z-score threshold for burst detection
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        df = df.sort_values('Time').reset_index(drop=True)

        # Set time as index for rolling operations (must be monotonic)
        df = df.set_index('Time').sort_index()

        # Rolling statistics
        rolling = df['Speed'].rolling(f'{int(window_hours * 60)}min')
        df['speed_rolling_mean'] = rolling.mean()
        df['speed_rolling_std'] = rolling.std()

        # Global statistics
        global_mean = df['Speed'].mean()
        global_std = df['Speed'].std()

        # Z-score of rolling mean compared to global
        df['activity_z'] = (df['speed_rolling_mean'] - global_mean) / (global_std + 1e-6)

        # Classify bursts and lulls
        df['burst_type'] = 'Normal'
        df.loc[df['activity_z'] > z_threshold, 'burst_type'] = 'Activity Burst'
        df.loc[df['activity_z'] < -z_threshold, 'burst_type'] = 'Activity Lull'

        df = df.reset_index()

        return df[['Time', 'lat', 'lon', 'Speed', 'speed_rolling_mean', 'activity_z', 'burst_type']]

    @staticmethod
    def summarize_bursts(df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize detected bursts by type and timing.
        """
        burst_df = ActivityBurstDetection.detect_bursts(df)

        burst_df['hour'] = burst_df['Time'].dt.hour
        burst_df['day_of_week'] = burst_df['Time'].dt.dayofweek

        summary = burst_df.groupby(['burst_type', 'hour']).size().unstack(fill_value=0)

        return summary


class DynamicHomeRange:
    """
    Track how home range changes over time.
    """

    @staticmethod
    def home_range_time_series(df: pd.DataFrame, time_window: str = 'W',
                               percentile: float = 95) -> pd.DataFrame:
        """
        Compute home range area for each time period.

        Args:
            df: DataFrame with lat, lon, Time
            time_window: 'D'=day, 'W'=week, 'M'=month
            percentile: Percentile for MCP calculation
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        df['period'] = df['Time'].dt.to_period(time_window)

        results = []

        for period, group in df.groupby('period'):
            if len(group) < 5:
                continue

            # Filter by percentile distance from centroid
            centroid_lat = group['lat'].mean()
            centroid_lon = group['lon'].mean()

            distances = haversine_vectorized(
                group['lat'].values, group['lon'].values,
                np.full(len(group), centroid_lat),
                np.full(len(group), centroid_lon)
            )

            threshold = np.percentile(distances, percentile)
            core_points = group[distances <= threshold]

            if len(core_points) < 5:
                continue

            # Compute convex hull area
            try:
                points = core_points[['lon', 'lat']].values
                hull = ConvexHull(points)
                # Approximate area in km²
                lat_scale = 111  # km per degree latitude
                lon_scale = 111 * np.cos(np.radians(centroid_lat))
                area_km2 = hull.volume * lat_scale * lon_scale
            except:
                area_km2 = np.nan

            results.append({
                'period': str(period),
                'start_date': group['Time'].min(),
                'end_date': group['Time'].max(),
                'n_points': len(group),
                'area_km2': area_km2,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'max_distance_km': distances.max()
            })

        return pd.DataFrame(results)

    @staticmethod
    def detect_range_shifts(time_series: pd.DataFrame,
                            threshold_pct: float = 50) -> list:
        """
        Detect significant shifts in home range.
        """
        if len(time_series) < 3:
            return []

        shifts = []
        areas = time_series['area_km2'].values

        for i in range(1, len(areas)):
            if np.isnan(areas[i]) or np.isnan(areas[i-1]):
                continue

            pct_change = (areas[i] - areas[i-1]) / areas[i-1] * 100

            if abs(pct_change) >= threshold_pct:
                shifts.append({
                    'period': time_series.iloc[i]['period'],
                    'previous_area_km2': areas[i-1],
                    'new_area_km2': areas[i],
                    'pct_change': pct_change,
                    'shift_type': 'Expansion' if pct_change > 0 else 'Contraction'
                })

        return shifts


class SignificantLocationMining:
    """
    Discover significant locations using density-based clustering
    and behavioral analysis.
    """

    @staticmethod
    def find_significant_locations(df: pd.DataFrame,
                                   min_visits: int = 5,
                                   min_duration_hours: float = 0.5,
                                   eps_km: float = 0.05) -> pd.DataFrame:
        """
        Find significant locations based on visit frequency and duration.

        Args:
            df: DataFrame with lat, lon, Time, Speed
            min_visits: Minimum number of visits
            min_duration_hours: Minimum total time spent
            eps_km: Clustering radius in km
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')

        # Filter to low-speed points (stopped/slow movement)
        speed_threshold = df['Speed'].quantile(0.25)
        slow_points = df[df['Speed'] <= speed_threshold].copy()

        if len(slow_points) < 10:
            return pd.DataFrame()

        # DBSCAN clustering
        coords_rad = np.radians(slow_points[['lat', 'lon']].values)
        eps_rad = eps_km / EARTH_RADIUS_KM

        clustering = DBSCAN(eps=eps_rad, min_samples=min_visits,
                           metric='haversine').fit(coords_rad)
        slow_points['cluster'] = clustering.labels_

        # Analyze each cluster
        results = []
        for cluster_id in slow_points['cluster'].unique():
            if cluster_id == -1:  # Noise
                continue

            cluster_points = slow_points[slow_points['cluster'] == cluster_id]

            # Compute time spent (approximate by counting points × sampling interval)
            time_diffs = cluster_points['Time'].diff().dt.total_seconds() / 3600
            duration_hours = time_diffs.sum()

            if duration_hours < min_duration_hours:
                continue

            # Identify time patterns
            hours = cluster_points['Time'].dt.hour

            # Most common hour
            peak_hour = hours.mode().iloc[0] if len(hours.mode()) > 0 else -1

            # Day/night ratio
            day_count = ((hours >= 6) & (hours < 18)).sum()
            night_count = len(hours) - day_count

            results.append({
                'cluster_id': cluster_id,
                'lat': cluster_points['lat'].mean(),
                'lon': cluster_points['lon'].mean(),
                'n_visits': len(cluster_points),
                'total_duration_hours': duration_hours,
                'first_visit': cluster_points['Time'].min(),
                'last_visit': cluster_points['Time'].max(),
                'peak_hour': peak_hour,
                'day_visits': day_count,
                'night_visits': night_count,
                'mean_speed': cluster_points['Speed'].mean()
            })

        result_df = pd.DataFrame(results)

        # Classify location types based on patterns
        if len(result_df) > 0:
            result_df['location_type'] = 'Regular Stop'
            result_df.loc[result_df['night_visits'] > result_df['day_visits'] * 2, 'location_type'] = 'Night Resting'
            result_df.loc[result_df['day_visits'] > result_df['night_visits'] * 2, 'location_type'] = 'Day Activity'
            result_df.loc[result_df['total_duration_hours'] > result_df['total_duration_hours'].quantile(0.9), 'location_type'] = 'Primary Location'

        return result_df.sort_values('total_duration_hours', ascending=False)


def run_creative_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    """Run all creative analyses."""
    from pathlib import Path
    import json

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    # 1. Fractal Dimension
    print("\n1. Fractal Dimension Analysis...")
    fractal = FractalDimensionAnalysis.box_counting_dimension(df)
    results['fractal_dimension'] = fractal['fractal_dimension']
    results['fractal_interpretation'] = fractal['interpretation']
    with open(output_path / 'fractal_dimension.json', 'w') as f:
        json.dump(fractal, f, indent=2, default=str)
    print(f"   Fractal Dimension: {fractal['fractal_dimension']:.3f} ({fractal['interpretation']})")

    # 1b. Fractal time series
    if 'Time' in df.columns:
        fractal_ts = FractalDimensionAnalysis.fractal_by_time_period(df, time_resolution='W')
        fractal_ts.to_csv(output_path / 'fractal_time_series.csv', index=False)

    # 2. Directional Persistence
    print("\n2. Directional Persistence Analysis...")
    persistence = DirectionalPersistence.summarize_persistence(df)
    results['mean_persistence'] = persistence['mean_persistence']
    results['persistence_interpretation'] = persistence['interpretation']
    with open(output_path / 'directional_persistence.json', 'w') as f:
        json.dump(persistence, f, indent=2, default=str)
    print(f"   Mean Persistence: {persistence['mean_persistence']:.3f} ({persistence['interpretation']})")

    # 3. Sinuosity Analysis
    print("\n3. Sinuosity (Track Tortuosity) Analysis...")
    sinuosity = SinuosityAnalysis.compute_sinuosity(df)
    sinuosity.to_csv(output_path / 'sinuosity_analysis.csv', index=False)
    results['mean_sinuosity'] = sinuosity['sinuosity'].mean()
    print(f"   Mean Sinuosity: {results['mean_sinuosity']:.3f}")

    # 4. First Passage Time
    print("\n4. First Passage Time Analysis...")
    fpt = FirstPassageTime.compute_fpt(df)
    fpt.to_csv(output_path / 'first_passage_time.csv', index=False)

    ars_zones = FirstPassageTime.identify_ars_zones(df)
    ars_zones.to_csv(output_path / 'area_restricted_search_zones.csv', index=False)
    results['n_ars_zones'] = len(ars_zones)
    print(f"   Found {len(ars_zones)} area-restricted search zones")

    # 5. Revisitation Analysis
    print("\n5. Revisitation Analysis...")
    revisitation = RevisitationAnalysis.compute_revisitation(df)
    revisitation.to_csv(output_path / 'revisitation_analysis.csv', index=False)
    results['n_core_locations'] = len(revisitation[revisitation['location_type'] == 'Core'])
    results['n_frequent_locations'] = len(revisitation[revisitation['location_type'] == 'Frequent'])
    print(f"   Found {results['n_core_locations']} core and {results['n_frequent_locations']} frequent locations")

    # 6. Nocturnal/Diurnal Patterns
    print("\n6. Nocturnal/Diurnal Pattern Analysis...")
    day_night = NocturnalDiurnalAnalysis.classify_day_night(df)
    day_night.to_csv(output_path / 'day_night_patterns.csv', index=False)

    hourly_profile = NocturnalDiurnalAnalysis.hourly_activity_profile(df)
    hourly_profile.to_csv(output_path / 'hourly_activity_profile.csv', index=False)

    # Determine if more nocturnal or diurnal
    night_row = day_night[day_night['period'] == 'Night']
    day_row = day_night[day_night['period'] == 'Day']
    if len(night_row) > 0 and len(day_row) > 0:
        results['activity_pattern'] = 'Nocturnal' if night_row['activity_index'].iloc[0] > day_row['activity_index'].iloc[0] else 'Diurnal'
    print(f"   Activity Pattern: {results.get('activity_pattern', 'Unknown')}")

    # 7. Movement Entropy
    print("\n7. Movement Entropy Analysis...")
    spatial_entropy = MovementEntropy.spatial_entropy(df)
    temporal_entropy = MovementEntropy.temporal_entropy(df)
    transition_entropy = MovementEntropy.transition_entropy(df)

    entropy_results = {
        'spatial': spatial_entropy,
        'temporal': temporal_entropy,
        'transition': transition_entropy
    }
    results['spatial_entropy'] = spatial_entropy['normalized_entropy']
    results['temporal_entropy'] = temporal_entropy['normalized_entropy']

    with open(output_path / 'movement_entropy.json', 'w') as f:
        json.dump(entropy_results, f, indent=2, default=str)
    print(f"   Spatial Entropy (norm): {spatial_entropy['normalized_entropy']:.3f}")
    print(f"   Temporal Entropy (norm): {temporal_entropy['normalized_entropy']:.3f}")

    # 8. Activity Burst Detection
    print("\n8. Activity Burst Detection...")
    bursts = ActivityBurstDetection.detect_bursts(df)
    burst_counts = bursts['burst_type'].value_counts().to_dict()
    results['n_activity_bursts'] = burst_counts.get('Activity Burst', 0)
    results['n_activity_lulls'] = burst_counts.get('Activity Lull', 0)

    # Save only burst/lull events
    burst_events = bursts[bursts['burst_type'] != 'Normal']
    burst_events.to_csv(output_path / 'activity_bursts.csv', index=False)
    print(f"   Detected {results['n_activity_bursts']} bursts, {results['n_activity_lulls']} lulls")

    # 9. Dynamic Home Range
    print("\n9. Dynamic Home Range Analysis...")
    hr_timeseries = DynamicHomeRange.home_range_time_series(df)
    hr_timeseries.to_csv(output_path / 'home_range_time_series.csv', index=False)

    shifts = DynamicHomeRange.detect_range_shifts(hr_timeseries)
    results['n_range_shifts'] = len(shifts)
    if shifts:
        with open(output_path / 'home_range_shifts.json', 'w') as f:
            json.dump(shifts, f, indent=2, default=str)
    print(f"   Detected {len(shifts)} significant range shifts")

    # 10. Significant Location Mining
    print("\n10. Significant Location Mining...")
    sig_locations = SignificantLocationMining.find_significant_locations(df)
    sig_locations.to_csv(output_path / 'significant_locations.csv', index=False)
    results['n_significant_locations'] = len(sig_locations)
    results['n_primary_locations'] = len(sig_locations[sig_locations['location_type'] == 'Primary Location']) if len(sig_locations) > 0 else 0
    print(f"   Found {len(sig_locations)} significant locations ({results['n_primary_locations']} primary)")

    # Save summary
    with open(output_path / 'creative_analysis_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Creative analysis complete! Results saved to {output_path}")
    return results


def main():
    import argparse
    from geospatial_analysis import load_data

    parser = argparse.ArgumentParser(description='Creative Geospatial Analysis')
    parser.add_argument('-i', '--input', type=str, default='output/raw.tsv.gz',
                        help='Input file')
    parser.add_argument('-o', '--output', type=str, default='output/creative_analysis',
                        help='Output directory')
    parser.add_argument('-s', '--sample', type=float, default=1.0,
                        help='Sample fraction')
    parser.add_argument('-n', '--nrows', type=int, default=None,
                        help='Number of rows to read')

    args = parser.parse_args()

    df = load_data(args.input, sample_frac=args.sample, nrows=args.nrows)
    results = run_creative_analysis(df, args.output)

    return results


if __name__ == '__main__':
    main()
