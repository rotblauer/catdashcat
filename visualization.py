"""
Visualization Module for CatDash Geospatial Analysis
=====================================================
Efficient plotting methods for large geospatial datasets.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# Use a non-interactive backend for server-side rendering
plt.switch_backend('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def datashade_scatter(ax, x: np.ndarray, y: np.ndarray,
                      resolution: int = 500, cmap: str = 'viridis') -> None:
    """
    Create a datashaded scatter plot for large datasets.
    Uses 2D histogram for efficient rendering.
    """
    # Compute 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=resolution)

    # Log scale for better visibility
    heatmap = np.log1p(heatmap)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto',
              cmap=cmap, interpolation='gaussian')


def plot_clusters_map(df: pd.DataFrame, output_path: Path,
                      cluster_col: str = 'cluster',
                      alpha: float = 0.1,
                      figsize: Tuple[int, int] = (14, 10)) -> None:
    """Plot clustered points on a map-like scatter."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Geographic view
    ax = axes[0]
    n_clusters = df[cluster_col].nunique()
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_clusters, 20)))

    for i, cluster_id in enumerate(df[cluster_col].unique()[:20]):
        mask = df[cluster_col] == cluster_id
        ax.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'],
                  c=[colors[i % 20]], alpha=alpha, s=1, label=f'C{cluster_id}')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Clusters (Geographic)')

    # Density heatmap
    ax = axes[1]
    datashade_scatter(ax, df['lon'].values, df['lat'].values, resolution=300)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Point Density')

    plt.tight_layout()
    plt.savefig(output_path / 'clusters_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: clusters_map.png")


def plot_hotspots(hotspots_df: pd.DataFrame, output_path: Path,
                  figsize: Tuple[int, int] = (12, 10)) -> None:
    """Plot hotspot analysis results."""
    fig, ax = plt.subplots(figsize=figsize)

    # Color map for hotspot categories
    color_map = {
        'Hot99': '#d73027',   # Dark red
        'Hot95': '#fc8d59',   # Light red
        'NotSig': '#ffffbf',  # Yellow/neutral
        'Cold95': '#91bfdb',  # Light blue
        'Cold99': '#4575b4',  # Dark blue
    }

    for hotspot_type, color in color_map.items():
        mask = hotspots_df['hotspot'] == hotspot_type
        if mask.sum() > 0:
            ax.scatter(hotspots_df.loc[mask, 'lon'],
                      hotspots_df.loc[mask, 'lat'],
                      c=color, s=hotspots_df.loc[mask, 'count'] / hotspots_df['count'].max() * 100,
                      alpha=0.6, label=hotspot_type, edgecolors='none')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Hotspots (Getis-Ord Gi*)')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path / 'hotspots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: hotspots.png")


def plot_speed_stats(speed_stats_df: pd.DataFrame, output_path: Path,
                     figsize: Tuple[int, int] = (14, 10)) -> None:
    """Plot speed statistics."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Hourly speed
    ax = axes[0, 0]
    hourly = speed_stats_df[speed_stats_df['group_type'] == 'hour']
    if len(hourly) > 0:
        ax.bar(hourly['group_value'], hourly['mean'], color='steelblue', alpha=0.7)
        ax.fill_between(hourly['group_value'],
                       hourly['mean'] - hourly['std'],
                       hourly['mean'] + hourly['std'],
                       alpha=0.3)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Mean Speed')
        ax.set_title('Speed by Hour')
        ax.set_xticks(range(0, 24, 3))

    # Day of week speed
    ax = axes[0, 1]
    daily = speed_stats_df[speed_stats_df['group_type'] == 'day_of_week']
    if len(daily) > 0:
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax.bar(range(7), daily['mean'].values[:7], color='coral', alpha=0.7)
        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Mean Speed')
        ax.set_title('Speed by Day of Week')

    # Activity speed
    ax = axes[1, 0]
    activity = speed_stats_df[speed_stats_df['group_type'] == 'activity']
    if len(activity) > 0:
        activity_sorted = activity.sort_values('mean', ascending=True)
        y_pos = range(len(activity_sorted))
        ax.barh(y_pos, activity_sorted['mean'], color='seagreen', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(activity_sorted['group_value'])
        ax.set_xlabel('Mean Speed')
        ax.set_title('Speed by Activity')

    # Speed distribution histogram (from hourly data as proxy)
    ax = axes[1, 1]
    if len(hourly) > 0:
        ax.hist(hourly['mean'], bins=20, color='purple', alpha=0.7, edgecolor='white')
        ax.axvline(hourly['mean'].mean(), color='red', linestyle='--', label='Mean')
        ax.set_xlabel('Speed')
        ax.set_ylabel('Frequency')
        ax.set_title('Speed Distribution (Hourly Means)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path / 'speed_stats.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: speed_stats.png")


def plot_activity_rhythm(rhythm_df: pd.DataFrame, output_path: Path,
                         figsize: Tuple[int, int] = (12, 5)) -> None:
    """Plot activity rhythm (circadian pattern)."""
    fig, ax = plt.subplots(figsize=figsize)

    hours = rhythm_df.index.values
    activity = rhythm_df['activity_index'].values

    # Create polar-like visualization
    ax.fill_between(hours, 0, activity, alpha=0.4, color='steelblue')
    ax.plot(hours, activity, color='steelblue', linewidth=2)

    # Mark dawn/dusk
    ax.axvspan(5, 7, alpha=0.2, color='orange', label='Dawn')
    ax.axvspan(18, 20, alpha=0.2, color='purple', label='Dusk')

    ax.set_xlim(0, 23)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Activity Index')
    ax.set_title('Daily Activity Rhythm')
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path / 'activity_rhythm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: activity_rhythm.png")


def plot_hourly_heatmap(matrix_df: pd.DataFrame, output_path: Path,
                        figsize: Tuple[int, int] = (14, 6)) -> None:
    """Plot hour x day-of-week activity heatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    im = ax.imshow(matrix_df.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(days)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    ax.set_title('Activity Heatmap (Hour × Day)')

    plt.colorbar(im, ax=ax, label='Relative Activity')

    plt.tight_layout()
    plt.savefig(output_path / 'hourly_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: hourly_heatmap.png")


def plot_seasonal_patterns(seasonal_df: pd.DataFrame, output_path: Path,
                           figsize: Tuple[int, int] = (10, 8)) -> None:
    """Plot seasonal movement patterns."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    colors = ['#2166ac', '#4dac26', '#d73027', '#f4a582']

    # Order by season
    seasonal_df = seasonal_df.reindex(seasons)

    # Speed by season
    ax = axes[0, 0]
    ax.bar(seasons, seasonal_df['speed_mean'], color=colors, alpha=0.7)
    ax.set_ylabel('Mean Speed')
    ax.set_title('Speed by Season')

    # Activity count by season
    ax = axes[0, 1]
    ax.bar(seasons, seasonal_df['count'], color=colors, alpha=0.7)
    ax.set_ylabel('Number of Records')
    ax.set_title('Activity Count by Season')

    # Spatial spread by season
    ax = axes[1, 0]
    spread = np.sqrt(seasonal_df['lat_spread']**2 + seasonal_df['lon_spread']**2)
    ax.bar(seasons, spread, color=colors, alpha=0.7)
    ax.set_ylabel('Spatial Spread (std)')
    ax.set_title('Geographic Spread by Season')

    # Pie chart of activity distribution
    ax = axes[1, 1]
    ax.pie(seasonal_df['count'], labels=seasons, colors=colors,
           autopct='%1.1f%%', startangle=90)
    ax.set_title('Activity Distribution')

    plt.tight_layout()
    plt.savefig(output_path / 'seasonal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: seasonal_patterns.png")


def plot_geohash_grid(geohash_df: pd.DataFrame, output_path: Path,
                      figsize: Tuple[int, int] = (14, 10)) -> None:
    """Plot geohash grid aggregation."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Count heatmap
    ax = axes[0]
    scatter = ax.scatter(geohash_df['lon_mean'], geohash_df['lat_mean'],
                        c=np.log1p(geohash_df['count']),
                        cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Log(Count)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Activity Density by Geohash')

    # Speed heatmap
    ax = axes[1]
    if 'Speed_mean' in geohash_df.columns:
        scatter = ax.scatter(geohash_df['lon_mean'], geohash_df['lat_mean'],
                            c=geohash_df['Speed_mean'],
                            cmap='plasma', s=5, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Mean Speed')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Mean Speed by Geohash')

    plt.tight_layout()
    plt.savefig(output_path / 'geohash_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: geohash_grid.png")


def plot_umap_analysis(df: pd.DataFrame, output_path: Path,
                       figsize: Tuple[int, int] = (16, 12)) -> None:
    """Plot UMAP analysis if umap columns exist."""
    if 'umap_1' not in df.columns or 'umap_2' not in df.columns:
        print("  Skipping UMAP plots (no umap columns found)")
        return

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # UMAP colored by Speed
    ax = axes[0, 0]
    sample = df.sample(n=min(50000, len(df)), random_state=42)
    scatter = ax.scatter(sample['umap_2'], sample['umap_1'],
                        c=sample['Speed'], cmap='Spectral_r',
                        s=1, alpha=0.3, vmin=0, vmax=sample['Speed'].quantile(0.95))
    plt.colorbar(scatter, ax=ax, label='Speed')
    ax.set_xlabel('UMAP 2')
    ax.set_ylabel('UMAP 1')
    ax.set_title('UMAP colored by Speed')

    # UMAP colored by Activity
    ax = axes[0, 1]
    if 'Activity' in df.columns:
        activities = sample['Activity'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(activities)))
        for i, activity in enumerate(activities):
            mask = sample['Activity'] == activity
            ax.scatter(sample.loc[mask, 'umap_2'], sample.loc[mask, 'umap_1'],
                      c=[colors[i]], s=1, alpha=0.3, label=activity)
        ax.legend(markerscale=5, loc='best')
    ax.set_xlabel('UMAP 2')
    ax.set_ylabel('UMAP 1')
    ax.set_title('UMAP colored by Activity')

    # UMAP density
    ax = axes[1, 0]
    datashade_scatter(ax, sample['umap_2'].values, sample['umap_1'].values,
                     resolution=200, cmap='magma')
    ax.set_xlabel('UMAP 2')
    ax.set_ylabel('UMAP 1')
    ax.set_title('UMAP Point Density')

    # UMAP colored by Name
    ax = axes[1, 1]
    if 'Name' in df.columns:
        names = sample['Name'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(names), 10)))
        for i, name in enumerate(names[:10]):
            mask = sample['Name'] == name
            ax.scatter(sample.loc[mask, 'umap_2'], sample.loc[mask, 'umap_1'],
                      c=[colors[i]], s=1, alpha=0.3, label=name)
        ax.legend(markerscale=5, loc='best')
    ax.set_xlabel('UMAP 2')
    ax.set_ylabel('UMAP 1')
    ax.set_title('UMAP colored by Name')

    plt.tight_layout()
    plt.savefig(output_path / 'umap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: umap_analysis.png")


def plot_home_range(df: pd.DataFrame, output_path: Path,
                    figsize: Tuple[int, int] = (10, 10)) -> None:
    """Plot home range visualization."""
    from scipy.spatial import ConvexHull

    fig, ax = plt.subplots(figsize=figsize)

    # Sample points for plotting
    sample = df.sample(n=min(50000, len(df)), random_state=42)
    ax.scatter(sample['lon'], sample['lat'], c='steelblue', s=1, alpha=0.1)

    # Compute and plot 95% MCP
    centroid_lat = df['lat'].mean()
    centroid_lon = df['lon'].mean()

    # Distance from centroid
    from geospatial_analysis import haversine_vectorized
    distances = haversine_vectorized(
        df['lat'].values, df['lon'].values,
        np.full(len(df), centroid_lat),
        np.full(len(df), centroid_lon)
    )

    for pct, color in [(95, 'red'), (75, 'orange'), (50, 'yellow')]:
        threshold = np.percentile(distances, pct)
        mask = distances <= threshold
        filtered = df[mask]

        if len(filtered) >= 3:
            try:
                points = filtered[['lon', 'lat']].values
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])  # Close polygon
                ax.plot(hull_points[:, 0], hull_points[:, 1],
                       color=color, linewidth=2, label=f'{pct}% MCP')
            except:
                pass

    ax.plot(centroid_lon, centroid_lat, 'k*', markersize=15, label='Centroid')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Home Range (Minimum Convex Polygon)')
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(output_path / 'home_range.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: home_range.png")


def create_summary_dashboard(output_path: Path,
                             figsize: Tuple[int, int] = (20, 16)) -> None:
    """Create a summary dashboard combining key visualizations."""
    # Load summary data
    summary_file = output_path / 'analysis_summary.json'
    if not summary_file.exists():
        print("  No summary file found, skipping dashboard")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    fig = plt.figure(figsize=figsize)

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Text summary
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    summary_text = "\n".join([
        "Analysis Summary",
        "=" * 30,
        f"Clusters: {summary.get('n_clusters', 'N/A')}",
        f"Hotspots: {summary.get('n_hotspots', 'N/A')}",
        f"Coldspots: {summary.get('n_coldspots', 'N/A')}",
        f"Segments: {summary.get('n_segments', 'N/A')}",
        f"Home Range: {summary.get('home_range_km2', 0):.2f} km²",
        f"Moran's I: {summary.get('morans_i', 0):.4f}",
        f"Pattern: {summary.get('morans_interpretation', 'N/A')}",
    ])
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    # Load and embed other plots if they exist
    plot_files = [
        ('clusters_map.png', gs[0, 1:3]),
        ('hotspots.png', gs[0, 3]),
        ('speed_stats.png', gs[1, :2]),
        ('hourly_heatmap.png', gs[1, 2:]),
        ('geohash_grid.png', gs[2, :2]),
        ('home_range.png', gs[2, 2:]),
    ]

    for filename, grid_pos in plot_files:
        filepath = output_path / filename
        if filepath.exists():
            ax = fig.add_subplot(grid_pos)
            img = plt.imread(filepath)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(filename.replace('.png', '').replace('_', ' ').title())

    plt.savefig(output_path / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: dashboard.png")


def generate_all_visualizations(analysis_dir: str, data_file: Optional[str] = None):
    """Generate all visualizations from analysis results."""
    output_path = Path(analysis_dir)

    if not output_path.exists():
        print(f"Analysis directory {analysis_dir} not found")
        return

    print("Generating visualizations...")

    # Load enriched data if available
    enriched_file = output_path / 'enriched_data.csv.gz'
    df = None
    if enriched_file.exists():
        print("Loading enriched data...")
        df = pd.read_csv(enriched_file, compression='gzip')
    elif data_file:
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file, sep='\t', compression='gzip' if data_file.endswith('.gz') else None)

    # Generate plots from CSV outputs
    csv_files = {
        'hotspots.csv': plot_hotspots,
        'speed_stats.csv': plot_speed_stats,
        'activity_rhythm.csv': plot_activity_rhythm,
        'hourly_activity_matrix.csv': plot_hourly_heatmap,
        'seasonal_patterns.csv': plot_seasonal_patterns,
        'geohash_aggregation.csv': plot_geohash_grid,
    }

    for filename, plot_func in csv_files.items():
        filepath = output_path / filename
        if filepath.exists():
            try:
                csv_df = pd.read_csv(filepath, index_col=0 if 'matrix' in filename or 'rhythm' in filename or 'seasonal' in filename else None)
                plot_func(csv_df, output_path)
            except Exception as e:
                print(f"  Error plotting {filename}: {e}")

    # Plots that need raw data
    if df is not None:
        print("\nGenerating data-dependent visualizations...")
        try:
            plot_clusters_map(df, output_path)
        except Exception as e:
            print(f"  Error plotting clusters: {e}")

        try:
            plot_home_range(df, output_path)
        except Exception as e:
            print(f"  Error plotting home range: {e}")

        try:
            plot_umap_analysis(df, output_path)
        except Exception as e:
            print(f"  Error plotting UMAP: {e}")

    # Create summary dashboard
    print("\nCreating summary dashboard...")
    try:
        create_summary_dashboard(output_path)
    except Exception as e:
        print(f"  Error creating dashboard: {e}")

    print("\n✓ Visualization complete!")


def main():
    parser = argparse.ArgumentParser(description='Visualize CatDash Analysis Results')
    parser.add_argument('-i', '--input', type=str, default='output/analysis',
                        help='Analysis output directory')
    parser.add_argument('-d', '--data', type=str, default=None,
                        help='Optional: raw data file for additional plots')

    args = parser.parse_args()
    generate_all_visualizations(args.input, args.data)


if __name__ == '__main__':
    main()
