
# CatDash - Geospatial Analysis for Cat Tracking Data

Scalable geospatial analysis toolkit for millions of GPS tracking points. Implements both standard spatial analysis methods and creative/advanced techniques.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis suite with 10% sample
python run_analysis.py -i output/raw.tsv.gz -o output/results -s 0.1 --visualize

# Run only standard analysis
python run_analysis.py -i output/raw.tsv.gz -o output/results -a standard

# Run only advanced analysis
python run_analysis.py -i output/raw.tsv.gz -o output/results -a advanced

# Quick test with limited rows
python run_analysis.py -i output/raw.tsv.gz -o output/test -n 10000
```

## Analysis Methods

### Standard Analysis (`geospatial_analysis.py`)

| Method | Class | Description |
|--------|-------|-------------|
| **Spatial Clustering** | `SpatialClustering` | DBSCAN with haversine distance, Geohash + K-means |
| **Hotspot Analysis** | `HotspotAnalysis` | Kernel Density Estimation, Getis-Ord Gi* |
| **Movement Patterns** | `MovementPatterns` | Stay point detection, speed statistics, trajectory segmentation |
| **Home Range** | `HomeRangeEstimation` | Minimum Convex Polygon (MCP), Utilization Distribution |
| **Temporal Patterns** | `TemporalPatterns` | Hour×Day heatmaps, circadian rhythms, seasonal patterns |
| **Grid Aggregation** | `GridAggregation` | Geohash cells, hexagonal grids |
| **Spatial Statistics** | `SpatialStatistics` | Global Moran's I autocorrelation |

### Advanced Analysis (`advanced_analysis.py`)

| Method | Class | Description |
|--------|-------|-------------|
| **Space-Time Cube** | `SpaceTimeCube` | Temporal hotspot detection with Mann-Kendall trends |
| **Movement Network** | `MovementNetwork` | Transition graphs, hub/connector identification |
| **Behavioral States** | `BehavioralStateClassification` | GMM-based state classification (Resting, Foraging, Traveling) |
| **Anomaly Detection** | `AnomalyDetection` | Isolation Forest spatial anomalies, speed outliers |
| **Territory Analysis** | `TerritoryAnalysis` | Jaccard overlap, Voronoi tessellation |
| **Point Patterns** | `RipleysK` | Ripley's K function for clustering detection |
| **Flow Analysis** | `FlowAnalysis` | Origin-destination flows, corridor identification |

## Example Usage in Python

```python
from geospatial_analysis import load_data, SpatialClustering, HotspotAnalysis
from advanced_analysis import BehavioralStateClassification, MovementNetwork

# Load data
df = load_data('output/raw.tsv.gz', sample_frac=0.1)

# Clustering
labels, geohash_stats = SpatialClustering.geohash_kmeans(df, n_clusters=50)

# Hotspot analysis  
hotspots = HotspotAnalysis.getis_ord_gi(df, geohash_precision=6)

# Behavioral states
df_states = BehavioralStateClassification.classify_states(df, n_states=4)

# Movement network
nodes, edges = MovementNetwork.build_transition_network(df)
nodes = MovementNetwork.find_hubs_and_connectors(nodes, edges)
```

## Output Files

The analysis generates the following files:

### Standard Analysis (`output/standard/`)
- `geohash_clusters.csv` - Cluster assignments
- `hotspots.csv` - Getis-Ord Gi* hotspot results
- `speed_stats.csv` - Speed by hour/day/activity
- `hourly_activity_matrix.csv` - Hour × day-of-week activity
- `activity_rhythm.csv` - Circadian activity patterns
- `seasonal_patterns.csv` - Seasonal movement patterns
- `geohash_aggregation.csv` - Grid-aggregated statistics
- `analysis_summary.json` - Summary statistics
- `enriched_data.csv.gz` - Data with cluster labels

### Advanced Analysis (`output/advanced/`)
- `emerging_hotspots.csv` - Space-time emerging/diminishing hotspots
- `network_nodes.csv` / `network_edges.csv` - Movement network
- `behavioral_states.csv.gz` - Classified behavioral states
- `anomalies.csv.gz` - Detected anomalies
- `territory_overlap.csv` - Individual territory overlap
- `ripleys_k.json` - Point pattern analysis results
- `flow_matrix.csv` / `corridors.csv` - Movement flows

### Visualizations (with `--visualize` flag)
- `clusters_map.png` - Cluster visualization
- `hotspots.png` - Hotspot map
- `speed_stats.png` - Speed statistics
- `hourly_heatmap.png` - Activity heatmap
- `home_range.png` - MCP home range
- `dashboard.png` - Summary dashboard

## Data Format

Input: Tab-separated gzipped file with columns:
- `lat`, `lon` - Coordinates (required)
- `Time` - ISO timestamp (optional, enables temporal analysis)
- `Speed` - Movement speed (optional)
- `Activity` - Activity type (optional)
- `Name` - Individual identifier (optional, enables territory analysis)
- `Heading` - Movement direction (optional)

## Scalability

Designed for millions of data points:
- Geohash-based aggregation reduces memory usage
- Sampling options for quick exploration
- BallTree for efficient spatial queries
- Mini-batch algorithms for large datasets

## Cat UMAPs (Original Analysis)

![](docs/index_files/figure-html/unnamed-chunk-1-1.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-2.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-3.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-4.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-5.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-6.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-7.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-8.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-9.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-10.jpeg)<!-- -->![](docs/index_files/figure-html/unnamed-chunk-1-11.jpeg)<!-- -->
