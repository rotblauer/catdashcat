"""
Advanced Geospatial Analysis Module for CatDash
================================================
Creative and cutting-edge geospatial analysis methods.

Includes:
1. Space-time cube analysis
2. Movement network analysis
3. Habitat preference modeling
4. Anomaly detection in movement
5. Behavioral state classification
6. Territory overlap analysis
7. Flow mapping
8. Point pattern analysis (Ripley's K)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance, Voronoi
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

from geospatial_analysis import encode_geohash, haversine_vectorized, EARTH_RADIUS_KM


class SpaceTimeCube:
    """
    Space-Time Cube analysis for temporal hot spot detection.
    Identifies emerging, intensifying, and diminishing hotspots.
    """

    @staticmethod
    def create_cube(df: pd.DataFrame, spatial_resolution: int = 6,
                    temporal_resolution: str = 'W') -> pd.DataFrame:
        """
        Create space-time cube bins.

        Args:
            df: DataFrame with lat, lon, Time
            spatial_resolution: Geohash precision
            temporal_resolution: Time binning ('D'=day, 'W'=week, 'M'=month)
        """
        df = df.copy()
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], spatial_resolution), axis=1)
        df['time_bin'] = df['Time'].dt.to_period(temporal_resolution)

        # Aggregate into space-time bins
        cube = df.groupby(['geohash', 'time_bin']).agg({
            'lat': 'mean',
            'lon': 'mean',
            'Speed': ['mean', 'std', 'count']
        }).reset_index()
        cube.columns = ['geohash', 'time_bin', 'lat', 'lon', 'speed_mean', 'speed_std', 'count']

        return cube

    @staticmethod
    def detect_emerging_hotspots(cube: pd.DataFrame, min_trend_periods: int = 3) -> pd.DataFrame:
        """
        Detect emerging, persistent, and diminishing hotspots using Mann-Kendall trend.
        """
        results = []

        for geohash in cube['geohash'].unique():
            location_data = cube[cube['geohash'] == geohash].sort_values('time_bin')

            if len(location_data) < min_trend_periods:
                continue

            counts = location_data['count'].values

            # Mann-Kendall trend test
            n = len(counts)
            s = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    s += np.sign(counts[j] - counts[i])

            # Variance of S
            var_s = n * (n - 1) * (2 * n + 5) / 18

            # Z-score
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0

            # Classify hotspot type
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            if p_value < 0.05:
                if z > 0:
                    hotspot_type = 'Emerging' if counts[-1] > counts[0] else 'Persistent Hot'
                else:
                    hotspot_type = 'Diminishing' if counts[-1] < counts[0] else 'Persistent Cold'
            else:
                hotspot_type = 'No Pattern'

            results.append({
                'geohash': geohash,
                'lat': location_data['lat'].mean(),
                'lon': location_data['lon'].mean(),
                'trend_z': z,
                'p_value': p_value,
                'hotspot_type': hotspot_type,
                'mean_count': counts.mean(),
                'n_periods': n
            })

        return pd.DataFrame(results)


class MovementNetwork:
    """
    Network-based analysis of movement patterns.
    Treats locations as nodes and transitions as edges.
    """

    @staticmethod
    def build_transition_network(df: pd.DataFrame,
                                  geohash_precision: int = 5) -> tuple:
        """
        Build a directed network of location transitions.

        Returns:
            nodes: DataFrame of unique locations
            edges: DataFrame of transitions with weights
        """
        df = df.copy().sort_values('Time')
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], geohash_precision), axis=1)

        # Create transitions
        df['next_geohash'] = df['geohash'].shift(-1)
        df['same_entity'] = df['Name'].shift(-1) == df['Name'] if 'Name' in df.columns else True

        # Filter valid transitions
        transitions = df[df['same_entity'] & (df['geohash'] != df['next_geohash'])].copy()

        # Aggregate edges
        edges = transitions.groupby(['geohash', 'next_geohash']).agg({
            'lat': 'count'
        }).reset_index()
        edges.columns = ['source', 'target', 'weight']

        # Create nodes
        nodes = df.groupby('geohash').agg({
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()
        nodes['visits'] = df.groupby('geohash').size().values

        return nodes, edges

    @staticmethod
    def find_hubs_and_connectors(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
        """
        Identify hub locations (high centrality) and connector routes.
        """
        # Build adjacency matrix
        node_list = nodes['geohash'].tolist()
        node_idx = {g: i for i, g in enumerate(node_list)}
        n = len(node_list)

        # In-degree and out-degree
        in_degree = np.zeros(n)
        out_degree = np.zeros(n)
        weighted_in = np.zeros(n)
        weighted_out = np.zeros(n)

        for _, row in edges.iterrows():
            if row['source'] in node_idx and row['target'] in node_idx:
                src_idx = node_idx[row['source']]
                tgt_idx = node_idx[row['target']]
                out_degree[src_idx] += 1
                in_degree[tgt_idx] += 1
                weighted_out[src_idx] += row['weight']
                weighted_in[tgt_idx] += row['weight']

        nodes = nodes.copy()
        nodes['in_degree'] = in_degree
        nodes['out_degree'] = out_degree
        nodes['total_degree'] = in_degree + out_degree
        nodes['weighted_in'] = weighted_in
        nodes['weighted_out'] = weighted_out

        # Classify node types
        degree_threshold = np.percentile(nodes['total_degree'], 75)
        nodes['node_type'] = 'Regular'
        nodes.loc[nodes['total_degree'] >= degree_threshold, 'node_type'] = 'Hub'
        nodes.loc[nodes['in_degree'] > 2 * nodes['out_degree'], 'node_type'] = 'Destination'
        nodes.loc[nodes['out_degree'] > 2 * nodes['in_degree'], 'node_type'] = 'Origin'

        return nodes


class BehavioralStateClassification:
    """
    Classify movement into behavioral states using Hidden Markov-like approaches.
    States: Resting, Foraging/Exploring, Traveling, etc.
    """

    @staticmethod
    def extract_movement_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for behavioral classification.
        """
        df = df.copy().sort_values(['Name', 'Time'] if 'Name' in df.columns else 'Time')

        # Time differences
        df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
        df['time_diff'] = df.groupby('Name')['Time'].diff().dt.total_seconds() if 'Name' in df.columns else df['Time'].diff().dt.total_seconds()

        # Step length (distance between consecutive points)
        df['prev_lat'] = df.groupby('Name')['lat'].shift(1) if 'Name' in df.columns else df['lat'].shift(1)
        df['prev_lon'] = df.groupby('Name')['lon'].shift(1) if 'Name' in df.columns else df['lon'].shift(1)

        mask = ~(df['prev_lat'].isna())
        df.loc[mask, 'step_length'] = haversine_vectorized(
            df.loc[mask, 'lat'].values,
            df.loc[mask, 'lon'].values,
            df.loc[mask, 'prev_lat'].values,
            df.loc[mask, 'prev_lon'].values
        )

        # Turn angle
        df['prev_heading'] = df.groupby('Name')['Heading'].shift(1) if 'Name' in df.columns else df['Heading'].shift(1)
        df['turn_angle'] = np.abs(df['Heading'] - df['prev_heading'])
        df.loc[df['turn_angle'] > 180, 'turn_angle'] = 360 - df.loc[df['turn_angle'] > 180, 'turn_angle']

        # Rolling statistics
        window = 5
        df['speed_rolling_mean'] = df.groupby('Name')['Speed'].rolling(window, min_periods=1).mean().reset_index(drop=True) if 'Name' in df.columns else df['Speed'].rolling(window, min_periods=1).mean()
        df['speed_rolling_std'] = df.groupby('Name')['Speed'].rolling(window, min_periods=1).std().reset_index(drop=True) if 'Name' in df.columns else df['Speed'].rolling(window, min_periods=1).std()

        return df

    @staticmethod
    def classify_states(df: pd.DataFrame, n_states: int = 4) -> pd.DataFrame:
        """
        Classify movement into behavioral states using Gaussian Mixture Model.
        """
        df = BehavioralStateClassification.extract_movement_features(df)

        # Select features for classification
        feature_cols = ['Speed', 'step_length', 'turn_angle', 'speed_rolling_std']
        available_cols = [c for c in feature_cols if c in df.columns]

        # Prepare features
        features = df[available_cols].fillna(0)

        # Sample for fitting if too large
        sample_size = min(100000, len(features))
        sample_idx = np.random.choice(len(features), sample_size, replace=False)

        # Fit GMM
        scaler = StandardScaler()
        scaled_sample = scaler.fit_transform(features.iloc[sample_idx])

        gmm = GaussianMixture(n_components=n_states, random_state=42, n_init=3)
        gmm.fit(scaled_sample)

        # Predict all points
        scaled_all = scaler.transform(features)
        df['behavioral_state'] = gmm.predict(scaled_all)
        df['state_probability'] = gmm.predict_proba(scaled_all).max(axis=1)

        # Label states based on characteristics
        state_stats = df.groupby('behavioral_state')['Speed'].mean().sort_values()
        state_labels = {state_stats.index[0]: 'Resting'}

        if n_states >= 2:
            state_labels[state_stats.index[-1]] = 'Traveling'
        if n_states >= 3:
            state_labels[state_stats.index[1]] = 'Foraging'
        if n_states >= 4:
            state_labels[state_stats.index[-2]] = 'Exploring'

        for i in state_stats.index:
            if i not in state_labels:
                state_labels[i] = f'State_{i}'

        df['state_label'] = df['behavioral_state'].map(state_labels)

        return df


class AnomalyDetection:
    """
    Detect anomalous movement patterns.
    """

    @staticmethod
    def detect_spatial_anomalies(df: pd.DataFrame,
                                  contamination: float = 0.05) -> pd.DataFrame:
        """
        Detect spatially anomalous points using Isolation Forest.
        """
        features = df[['lat', 'lon', 'Speed']].copy()

        # Add time-based features if available
        if 'Time' in df.columns:
            df_time = pd.to_datetime(df['Time'], format='ISO8601')
            features['hour'] = df_time.dt.hour
            features['day_of_week'] = df_time.dt.dayofweek

        # Sample for fitting
        sample_size = min(50000, len(features))
        sample_idx = np.random.choice(len(features), sample_size, replace=False)

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination,
                                      random_state=42, n_jobs=-1)
        iso_forest.fit(features.iloc[sample_idx].fillna(0))

        # Predict anomalies
        df = df.copy()
        df['anomaly_score'] = iso_forest.decision_function(features.fillna(0))
        df['is_anomaly'] = iso_forest.predict(features.fillna(0)) == -1

        return df

    @staticmethod
    def detect_speed_anomalies(df: pd.DataFrame,
                                threshold_std: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalous speed values.
        """
        df = df.copy()

        # Per-activity anomaly detection
        if 'Activity' in df.columns:
            df['speed_z'] = df.groupby('Activity')['Speed'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )
        else:
            df['speed_z'] = (df['Speed'] - df['Speed'].mean()) / (df['Speed'].std() + 1e-6)

        df['speed_anomaly'] = np.abs(df['speed_z']) > threshold_std

        return df


class TerritoryAnalysis:
    """
    Analyze territory and space use patterns.
    """

    @staticmethod
    def compute_overlap(df: pd.DataFrame, geohash_precision: int = 6) -> pd.DataFrame:
        """
        Compute territory overlap between different individuals/names.
        """
        if 'Name' not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], geohash_precision), axis=1)

        # Get locations used by each individual
        individual_locations = df.groupby('Name')['geohash'].apply(set).to_dict()
        names = list(individual_locations.keys())

        # Compute pairwise overlap
        overlaps = []
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                locs1 = individual_locations[name1]
                locs2 = individual_locations[name2]

                intersection = len(locs1 & locs2)
                union = len(locs1 | locs2)

                jaccard = intersection / union if union > 0 else 0
                overlap_pct_1 = intersection / len(locs1) if len(locs1) > 0 else 0
                overlap_pct_2 = intersection / len(locs2) if len(locs2) > 0 else 0

                overlaps.append({
                    'individual_1': name1,
                    'individual_2': name2,
                    'shared_locations': intersection,
                    'jaccard_index': jaccard,
                    'overlap_pct_1': overlap_pct_1,
                    'overlap_pct_2': overlap_pct_2,
                    'total_1': len(locs1),
                    'total_2': len(locs2)
                })

        return pd.DataFrame(overlaps)

    @staticmethod
    def voronoi_territories(df: pd.DataFrame) -> dict:
        """
        Create Voronoi tessellation based on activity centers.
        """
        if 'Name' not in df.columns:
            return {}

        # Compute activity centers
        centers = df.groupby('Name').agg({
            'lat': 'mean',
            'lon': 'mean'
        }).reset_index()

        if len(centers) < 4:
            return {'centers': centers.to_dict('records')}

        points = centers[['lon', 'lat']].values

        try:
            vor = Voronoi(points)

            return {
                'centers': centers.to_dict('records'),
                'vertices': vor.vertices.tolist(),
                'regions': [list(r) for r in vor.regions],
                'point_region': vor.point_region.tolist()
            }
        except Exception as e:
            return {'centers': centers.to_dict('records'), 'error': str(e)}


class RipleysK:
    """
    Point pattern analysis using Ripley's K function.
    """

    @staticmethod
    def compute_k_function(df: pd.DataFrame, max_distance_km: float = 10,
                           n_distances: int = 50,
                           sample_size: int = 5000) -> dict:
        """
        Compute Ripley's K function to detect clustering/dispersion.

        K(r) > πr² indicates clustering
        K(r) < πr² indicates dispersion
        """
        # Sample for efficiency
        if len(df) > sample_size:
            sample = df.sample(n=sample_size, random_state=42)
        else:
            sample = df

        coords = np.radians(sample[['lat', 'lon']].values)
        n = len(coords)

        # Study area (approximate from data extent)
        lat_range = df['lat'].max() - df['lat'].min()
        lon_range = df['lon'].max() - df['lon'].min()
        # Convert to km
        area_km2 = lat_range * 111 * lon_range * 111 * np.cos(np.radians(df['lat'].mean()))

        # Distance range
        distances = np.linspace(0.1, max_distance_km, n_distances)
        k_values = []

        # Build tree for efficient distance queries
        from sklearn.neighbors import BallTree
        tree = BallTree(coords, metric='haversine')

        for d in distances:
            d_rad = d / EARTH_RADIUS_KM
            counts = tree.query_radius(coords, r=d_rad, count_only=True)
            # Subtract self-counts
            mean_count = (counts.sum() - n) / n
            k = area_km2 * mean_count / (n - 1)
            k_values.append(k)

        # Compute L function (variance-stabilized K)
        l_values = np.sqrt(np.array(k_values) / np.pi) - distances

        # Expected under CSR (Complete Spatial Randomness)
        expected_k = np.pi * distances**2

        return {
            'distances': distances.tolist(),
            'k_values': k_values,
            'l_values': l_values.tolist(),
            'expected_k': expected_k.tolist(),
            'area_km2': area_km2,
            'n_points': n,
            'interpretation': 'Clustered' if np.mean(l_values) > 0 else 'Dispersed'
        }


class FlowAnalysis:
    """
    Analyze movement flows between locations.
    """

    @staticmethod
    def compute_flow_matrix(df: pd.DataFrame,
                            geohash_precision: int = 5) -> pd.DataFrame:
        """
        Compute origin-destination flow matrix.
        """
        df = df.copy().sort_values(['Name', 'Time'] if 'Name' in df.columns else 'Time')
        df['geohash'] = df.apply(lambda r: encode_geohash(r['lat'], r['lon'], geohash_precision), axis=1)

        # Get coordinates for each geohash
        geohash_coords = df.groupby('geohash').agg({'lat': 'mean', 'lon': 'mean'}).to_dict('index')

        # Compute flows
        df['next_geohash'] = df.groupby('Name')['geohash'].shift(-1) if 'Name' in df.columns else df['geohash'].shift(-1)

        # Filter to actual movements
        flows = df[df['geohash'] != df['next_geohash']].copy()
        flows = flows.dropna(subset=['next_geohash'])

        # Aggregate flows
        flow_matrix = flows.groupby(['geohash', 'next_geohash']).size().reset_index(name='flow_count')

        # Add coordinates
        flow_matrix['origin_lat'] = flow_matrix['geohash'].map(lambda x: geohash_coords.get(x, {}).get('lat'))
        flow_matrix['origin_lon'] = flow_matrix['geohash'].map(lambda x: geohash_coords.get(x, {}).get('lon'))
        flow_matrix['dest_lat'] = flow_matrix['next_geohash'].map(lambda x: geohash_coords.get(x, {}).get('lat'))
        flow_matrix['dest_lon'] = flow_matrix['next_geohash'].map(lambda x: geohash_coords.get(x, {}).get('lon'))

        return flow_matrix

    @staticmethod
    def identify_corridors(flow_matrix: pd.DataFrame,
                           percentile: float = 90) -> pd.DataFrame:
        """
        Identify high-traffic movement corridors.
        """
        threshold = np.percentile(flow_matrix['flow_count'], percentile)
        corridors = flow_matrix[flow_matrix['flow_count'] >= threshold].copy()
        corridors = corridors.sort_values('flow_count', ascending=False)

        return corridors


def run_advanced_analysis(df: pd.DataFrame, output_dir: str) -> dict:
    """Run all advanced analyses."""
    from pathlib import Path
    import json

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    results = {}

    # 1. Space-Time Cube
    print("\n1. Space-Time Cube Analysis...")
    if 'Time' in df.columns:
        cube = SpaceTimeCube.create_cube(df, temporal_resolution='W')
        emerging = SpaceTimeCube.detect_emerging_hotspots(cube)
        emerging.to_csv(output_path / 'emerging_hotspots.csv', index=False)
        results['emerging_hotspots'] = len(emerging[emerging['hotspot_type'] == 'Emerging'])
        results['diminishing_hotspots'] = len(emerging[emerging['hotspot_type'] == 'Diminishing'])
        print(f"   Found {results['emerging_hotspots']} emerging, {results['diminishing_hotspots']} diminishing hotspots")

    # 2. Movement Network
    print("\n2. Movement Network Analysis...")
    nodes, edges = MovementNetwork.build_transition_network(df)
    nodes = MovementNetwork.find_hubs_and_connectors(nodes, edges)
    nodes.to_csv(output_path / 'network_nodes.csv', index=False)
    edges.to_csv(output_path / 'network_edges.csv', index=False)
    results['n_network_hubs'] = len(nodes[nodes['node_type'] == 'Hub'])
    print(f"   Found {results['n_network_hubs']} hub locations")

    # 3. Behavioral State Classification
    print("\n3. Behavioral State Classification...")
    df_states = BehavioralStateClassification.classify_states(df, n_states=4)
    state_counts = df_states['state_label'].value_counts().to_dict()
    results['behavioral_states'] = state_counts
    df_states[['lat', 'lon', 'Speed', 'behavioral_state', 'state_label', 'state_probability']].to_csv(
        output_path / 'behavioral_states.csv.gz', compression='gzip', index=False
    )
    print(f"   State distribution: {state_counts}")

    # 4. Anomaly Detection
    print("\n4. Anomaly Detection...")
    df_anomalies = AnomalyDetection.detect_spatial_anomalies(df)
    results['n_anomalies'] = df_anomalies['is_anomaly'].sum()
    anomalies = df_anomalies[df_anomalies['is_anomaly']]
    if len(anomalies) > 0:
        anomalies.to_csv(output_path / 'anomalies.csv.gz', compression='gzip', index=False)
    print(f"   Detected {results['n_anomalies']} anomalous points")

    # 5. Territory Overlap
    print("\n5. Territory Overlap Analysis...")
    if 'Name' in df.columns and df['Name'].nunique() > 1:
        overlap = TerritoryAnalysis.compute_overlap(df)
        overlap.to_csv(output_path / 'territory_overlap.csv', index=False)
        if len(overlap) > 0:
            results['max_territory_overlap'] = overlap['jaccard_index'].max()
            print(f"   Max Jaccard overlap: {results['max_territory_overlap']:.3f}")

    # 6. Ripley's K
    print("\n6. Point Pattern Analysis (Ripley's K)...")
    ripleys = RipleysK.compute_k_function(df)
    results['point_pattern'] = ripleys['interpretation']
    with open(output_path / 'ripleys_k.json', 'w') as f:
        json.dump(ripleys, f, indent=2)
    print(f"   Pattern: {ripleys['interpretation']}")

    # 7. Flow Analysis
    print("\n7. Flow Analysis...")
    flow_matrix = FlowAnalysis.compute_flow_matrix(df)
    corridors = FlowAnalysis.identify_corridors(flow_matrix)
    flow_matrix.to_csv(output_path / 'flow_matrix.csv', index=False)
    corridors.to_csv(output_path / 'corridors.csv', index=False)
    results['n_corridors'] = len(corridors)
    print(f"   Identified {len(corridors)} high-traffic corridors")

    # Save summary
    with open(output_path / 'advanced_analysis_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Advanced analysis complete! Results saved to {output_path}")
    return results


def main():
    import argparse
    from geospatial_analysis import load_data

    parser = argparse.ArgumentParser(description='Advanced Geospatial Analysis')
    parser.add_argument('-i', '--input', type=str, default='output/raw.tsv.gz',
                        help='Input file')
    parser.add_argument('-o', '--output', type=str, default='output/advanced_analysis',
                        help='Output directory')
    parser.add_argument('-s', '--sample', type=float, default=1.0,
                        help='Sample fraction')
    parser.add_argument('-n', '--nrows', type=int, default=None,
                        help='Number of rows to read')

    args = parser.parse_args()

    df = load_data(args.input, sample_frac=args.sample, nrows=args.nrows)
    results = run_advanced_analysis(df, args.output)

    return results


if __name__ == '__main__':
    main()
