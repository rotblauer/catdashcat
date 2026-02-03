#!/usr/bin/env python3
"""
CatDash - Geospatial Analysis for Cat Tracking Data
====================================================

This module provides comprehensive geospatial analysis tools for analyzing
GPS tracking data with millions of data points.

Analysis Modules:
-----------------
- geospatial_analysis.py: Standard spatial analysis (clustering, hotspots, etc.)
- advanced_analysis.py: Creative analyses (networks, behavioral states, etc.)
- visualization.py: Efficient visualization for large datasets

Quick Start:
-----------
    # Run full analysis suite
    python run_analysis.py -i output/raw.tsv.gz -o output/results --visualize

    # Run with sampling for speed
    python run_analysis.py -i output/raw.tsv.gz -o output/results -s 0.1

    # Import for custom analysis
    from geospatial_analysis import SpatialClustering, HotspotAnalysis
    from advanced_analysis import BehavioralStateClassification, MovementNetwork
"""

import argparse
import sys


def main():
    """Entry point that delegates to run_analysis.py"""
    # Import and run the main analysis
    from run_analysis import main as run_main
    run_main()


def quick_analysis(input_file: str, output_dir: str = 'output/quick_analysis',
                   sample_frac: float = 0.1) -> dict:
    """
    Quick analysis function for interactive use.

    Args:
        input_file: Path to input TSV (gzipped)
        output_dir: Output directory
        sample_frac: Fraction of data to sample

    Returns:
        Dictionary of analysis results

    Example:
        >>> from main import quick_analysis
        >>> results = quick_analysis('output/raw.tsv.gz', sample_frac=0.1)
    """
    from geospatial_analysis import load_data, run_analysis

    df = load_data(input_file, sample_frac=sample_frac)
    results, enriched_df = run_analysis(df, output_dir)

    return results


def list_analyses():
    """Print available analysis methods."""
    print("""
CatDash Geospatial Analysis Methods
===================================

STANDARD ANALYSIS (geospatial_analysis.py):
-------------------------------------------
• SpatialClustering
  - dbscan_haversine: DBSCAN with haversine distance
  - geohash_kmeans: Two-stage geohash + K-means clustering

• HotspotAnalysis  
  - kernel_density_grid: KDE on spatial grid
  - getis_ord_gi: Getis-Ord Gi* statistic for hotspot detection

• MovementPatterns
  - detect_stay_points: Identify locations with extended stays
  - compute_speed_stats: Speed statistics by time/activity
  - trajectory_segmentation: Split trajectories by time gaps

• HomeRangeEstimation
  - minimum_convex_polygon: MCP at various percentiles
  - utilization_distribution: Kernel density contours

• TemporalPatterns
  - hourly_activity_matrix: Hour x day-of-week patterns
  - activity_rhythm: Circadian activity patterns
  - seasonal_patterns: Seasonal movement variations

• GridAggregation
  - geohash_aggregate: Aggregate by geohash cells
  - hex_grid_aggregate: Hexagonal grid aggregation

• SpatialStatistics
  - global_morans_i: Spatial autocorrelation

ADVANCED ANALYSIS (advanced_analysis.py):
-----------------------------------------
• SpaceTimeCube
  - create_cube: Build space-time bins
  - detect_emerging_hotspots: Mann-Kendall trend detection

• MovementNetwork
  - build_transition_network: Location transition graph
  - find_hubs_and_connectors: Network centrality analysis

• BehavioralStateClassification
  - extract_movement_features: Step length, turn angle, etc.
  - classify_states: GMM-based behavioral classification

• AnomalyDetection
  - detect_spatial_anomalies: Isolation Forest anomalies
  - detect_speed_anomalies: Statistical speed outliers

• TerritoryAnalysis
  - compute_overlap: Jaccard overlap between individuals
  - voronoi_territories: Voronoi tessellation

• RipleysK
  - compute_k_function: Point pattern analysis

• FlowAnalysis
  - compute_flow_matrix: Origin-destination flows
  - identify_corridors: High-traffic movement paths
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CatDash Geospatial Analysis')
    parser.add_argument('--list', action='store_true', help='List available analyses')

    args, remaining = parser.parse_known_args()

    if args.list:
        list_analyses()
    else:
        # Pass through to run_analysis
        sys.argv = [sys.argv[0]] + remaining
        main()
