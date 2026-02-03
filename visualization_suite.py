"""
Comprehensive Visualization Suite for CatDash
==============================================
Stunning visualizations for Standard, Advanced, and Creative analysis results.

Features:
- Publication-quality figures
- Interactive-style static plots
- Dark and light themes
- Comprehensive dashboards
- Geographic projections
- Animated-style time series
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
from pathlib import Path
import json
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Use non-interactive backend
plt.switch_backend('Agg')

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

# Custom color palettes
PALETTE_VIBRANT = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                   '#ffff33', '#a65628', '#f781bf', '#999999']
PALETTE_COOL = ['#1a9850', '#91cf60', '#d9ef8b', '#ffffbf', '#fee08b',
                '#fc8d59', '#d73027']
PALETTE_DARK = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']
PALETTE_CATEGORICAL = plt.cm.Set2.colors

# Custom colormaps
def create_density_cmap():
    """Create a beautiful density colormap."""
    colors = ['#0d0887', '#5302a3', '#8b0aa5', '#b83289', '#db5c68',
              '#f48849', '#febd2a', '#f0f921']
    return LinearSegmentedColormap.from_list('density', colors)

def create_diverging_cmap():
    """Create a diverging colormap for positive/negative values."""
    colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7',
              '#fddbc7', '#ef8a62', '#b2182b']
    return LinearSegmentedColormap.from_list('diverging', colors)

CMAP_DENSITY = create_density_cmap()
CMAP_DIVERGING = create_diverging_cmap()


def setup_style(dark_mode: bool = False):
    """Setup matplotlib style for stunning visuals."""
    if dark_mode:
        plt.style.use('dark_background')
        plt.rcParams.update({
            'figure.facecolor': '#1a1a2e',
            'axes.facecolor': '#16213e',
            'axes.edgecolor': '#e94560',
            'axes.labelcolor': '#eee',
            'text.color': '#eee',
            'xtick.color': '#eee',
            'ytick.color': '#eee',
            'grid.color': '#333',
        })
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': '#f8f9fa',
            'axes.edgecolor': '#333',
            'axes.labelcolor': '#333',
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
        })

setup_style(dark_mode=False)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_title_with_style(ax, title: str, subtitle: str = None):
    """Add a styled title with optional subtitle."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
                fontsize=9, color='gray', ha='center')


def create_gradient_fill(ax, x, y, color='steelblue', alpha=0.5):
    """Create a gradient fill under a line."""
    ax.fill_between(x, 0, y, alpha=alpha, color=color)
    ax.plot(x, y, color=color, linewidth=2)


def add_annotations(ax, x, y, labels, color='black'):
    """Add smart annotations avoiding overlap."""
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), fontsize=8, ha='center', va='bottom',
                   color=color, alpha=0.8)


# ============================================================================
# STANDARD ANALYSIS VISUALIZATIONS
# ============================================================================

def plot_spatial_overview(df: pd.DataFrame, output_path: Path,
                          figsize: Tuple[int, int] = (16, 12)) -> None:
    """Create a comprehensive spatial overview with multiple views."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.25)

    # Main density map
    ax1 = fig.add_subplot(gs[0, :2])
    heatmap, xedges, yedges = np.histogram2d(df['lon'], df['lat'], bins=300)
    heatmap = np.log1p(heatmap)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax1.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto',
                    cmap=CMAP_DENSITY, interpolation='gaussian')
    plt.colorbar(im, ax=ax1, label='Log Density', shrink=0.8)
    add_title_with_style(ax1, 'Spatial Density Map', 'Activity concentration patterns')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # Cluster visualization (if available)
    ax2 = fig.add_subplot(gs[0, 2])
    if 'cluster' in df.columns:
        sample = df.sample(n=min(20000, len(df)), random_state=42)
        n_clusters = min(sample['cluster'].nunique(), 10)
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
        for i, cluster_id in enumerate(sample['cluster'].unique()[:10]):
            mask = sample['cluster'] == cluster_id
            ax2.scatter(sample.loc[mask, 'lon'], sample.loc[mask, 'lat'],
                       c=[colors[i]], s=1, alpha=0.5)
        add_title_with_style(ax2, 'Cluster Distribution')
    else:
        sample = df.sample(n=min(20000, len(df)), random_state=42)
        ax2.scatter(sample['lon'], sample['lat'], c=sample['Speed'],
                   cmap='viridis', s=1, alpha=0.3)
        add_title_with_style(ax2, 'Speed Distribution')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    # Speed heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    speed_mean, _, _, _ = plt.mlab.griddata if hasattr(plt.mlab, 'griddata') else (None, None, None, None)
    # Use histogram2d for speed
    sample = df.sample(n=min(50000, len(df)), random_state=42)
    scatter = ax3.scatter(sample['lon'], sample['lat'], c=sample['Speed'],
                         cmap='plasma', s=2, alpha=0.4,
                         vmax=sample['Speed'].quantile(0.95))
    plt.colorbar(scatter, ax=ax3, label='Speed', shrink=0.8)
    add_title_with_style(ax3, 'Speed Spatial Distribution')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')

    # Temporal distribution
    ax4 = fig.add_subplot(gs[1, 1])
    if 'Time' in df.columns:
        df_temp = df.copy()
        df_temp['Time'] = pd.to_datetime(df_temp['Time'], format='ISO8601')
        df_temp['hour'] = df_temp['Time'].dt.hour
        hourly_counts = df_temp.groupby('hour').size()
        bars = ax4.bar(hourly_counts.index, hourly_counts.values,
                      color=plt.cm.twilight(np.linspace(0, 1, 24)),
                      edgecolor='white', linewidth=0.5)
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Number of Records')
        add_title_with_style(ax4, 'Temporal Distribution', '24-hour activity pattern')
        ax4.set_xticks(range(0, 24, 3))

    # Statistics summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    stats_text = f"""
    📊 Dataset Statistics
    ━━━━━━━━━━━━━━━━━━━━
    
    Total Records: {len(df):,}
    
    Spatial Extent:
    • Lat: {df['lat'].min():.4f} to {df['lat'].max():.4f}
    • Lon: {df['lon'].min():.4f} to {df['lon'].max():.4f}
    
    Speed Statistics:
    • Mean: {df['Speed'].mean():.2f}
    • Median: {df['Speed'].median():.2f}
    • Max: {df['Speed'].max():.2f}
    
    Unique Names: {df['Name'].nunique() if 'Name' in df.columns else 'N/A'}
    """
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))

    plt.savefig(output_path / 'spatial_overview.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ Saved: spatial_overview.png")


def plot_hotspots_advanced(hotspots_df: pd.DataFrame, output_path: Path,
                           figsize: Tuple[int, int] = (14, 10)) -> None:
    """Create an advanced hotspot visualization."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Spatial hotspot map
    ax = axes[0]

    # Background context
    ax.scatter(hotspots_df['lon'], hotspots_df['lat'], c='lightgray',
               s=5, alpha=0.3, label='All cells')

    # Hotspot styling
    hotspot_styles = {
        'Hot99': {'color': '#d73027', 'size': 120, 'marker': '*', 'label': 'Hot (99%)'},
        'Hot95': {'color': '#fc8d59', 'size': 80, 'marker': '^', 'label': 'Hot (95%)'},
        'Cold99': {'color': '#4575b4', 'size': 120, 'marker': '*', 'label': 'Cold (99%)'},
        'Cold95': {'color': '#91bfdb', 'size': 80, 'marker': 'v', 'label': 'Cold (95%)'},
    }

    for hotspot_type, style in hotspot_styles.items():
        mask = hotspots_df['hotspot'] == hotspot_type
        if mask.sum() > 0:
            ax.scatter(hotspots_df.loc[mask, 'lon'], hotspots_df.loc[mask, 'lat'],
                      c=style['color'], s=style['size'], marker=style['marker'][:1],
                      alpha=0.8, label=style['label'], edgecolors='white', linewidth=0.5)

    add_title_with_style(ax, 'Spatial Hotspot Analysis', 'Getis-Ord Gi* statistics')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='lower right', fontsize=9)

    # Right: Hotspot statistics
    ax = axes[1]
    categories = ['Hot99', 'Hot95', 'NotSig', 'Cold95', 'Cold99']
    counts = [len(hotspots_df[hotspots_df['hotspot'] == cat]) for cat in categories]
    colors = ['#d73027', '#fc8d59', '#ffffbf', '#91bfdb', '#4575b4']

    bars = ax.barh(categories, counts, color=colors, edgecolor='white', linewidth=2)
    ax.set_xlabel('Number of Cells')
    add_title_with_style(ax, 'Hotspot Distribution', 'Count by significance level')

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / 'hotspots_advanced.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: hotspots_advanced.png")


def plot_temporal_patterns(df: pd.DataFrame, output_path: Path,
                           figsize: Tuple[int, int] = (16, 12)) -> None:
    """Create comprehensive temporal pattern visualizations."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    df = df.copy()
    df['Time'] = pd.to_datetime(df['Time'], format='ISO8601')
    df['hour'] = df['Time'].dt.hour
    df['day_of_week'] = df['Time'].dt.dayofweek
    df['month'] = df['Time'].dt.month

    # 1. Circular hour plot (polar)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    hourly = df.groupby('hour').size()
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    radii = hourly.values
    width = 2 * np.pi / 24
    colors = plt.cm.twilight(np.linspace(0, 1, 24))
    bars = ax1.bar(theta, radii, width=width, color=colors, alpha=0.8, edgecolor='white')
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_xticks(theta)
    ax1.set_xticklabels([f'{h}h' for h in range(24)], fontsize=8)
    ax1.set_title('Activity by Hour\n(Polar View)', fontsize=12, fontweight='bold', pad=15)

    # 2. Day of week pattern
    ax2 = fig.add_subplot(gs[0, 1])
    daily = df.groupby('day_of_week').agg({'Speed': 'mean', 'lat': 'count'})
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    x = np.arange(7)

    ax2_twin = ax2.twinx()
    bars = ax2.bar(x - 0.2, daily['lat'], 0.4, color='steelblue', alpha=0.7, label='Activity Count')
    line = ax2_twin.plot(x + 0.2, daily['Speed'], 'ro-', markersize=8, linewidth=2, label='Mean Speed')

    ax2.set_xticks(x)
    ax2.set_xticklabels(days)
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Activity Count', color='steelblue')
    ax2_twin.set_ylabel('Mean Speed', color='red')
    add_title_with_style(ax2, 'Weekly Patterns')

    # 3. Hour x Day heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    pivot = df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
    im = ax3.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax3.set_xticks(range(0, 24, 3))
    ax3.set_xticklabels(range(0, 24, 3))
    ax3.set_yticks(range(7))
    ax3.set_yticklabels(days)
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Day')
    plt.colorbar(im, ax=ax3, shrink=0.8, label='Count')
    add_title_with_style(ax3, 'Activity Heatmap', 'Hour × Day of Week')

    # 4. Monthly trend
    ax4 = fig.add_subplot(gs[1, 0])
    monthly = df.groupby('month').agg({'Speed': 'mean', 'lat': 'count'})
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    x = monthly.index.values
    create_gradient_fill(ax4, x, monthly['lat'].values, color='steelblue', alpha=0.4)
    ax4.set_xticks(range(1, 13))
    ax4.set_xticklabels([months[i-1] for i in range(1, 13)], rotation=45)
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Activity Count')
    add_title_with_style(ax4, 'Seasonal Pattern')

    # 5. Speed by hour with confidence
    ax5 = fig.add_subplot(gs[1, 1])
    hourly_speed = df.groupby('hour')['Speed'].agg(['mean', 'std'])
    ax5.fill_between(hourly_speed.index,
                    hourly_speed['mean'] - hourly_speed['std'],
                    hourly_speed['mean'] + hourly_speed['std'],
                    alpha=0.3, color='coral')
    ax5.plot(hourly_speed.index, hourly_speed['mean'], 'o-', color='coral',
             linewidth=2, markersize=6)
    ax5.axhline(df['Speed'].mean(), linestyle='--', color='gray', alpha=0.5, label='Overall Mean')
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Speed (mean ± std)')
    ax5.set_xticks(range(0, 24, 3))
    add_title_with_style(ax5, 'Speed by Hour', 'With standard deviation')
    ax5.legend()

    # 6. Activity timeline (sample)
    ax6 = fig.add_subplot(gs[1, 2])
    sample = df.sample(n=min(5000, len(df)), random_state=42).sort_values('Time')
    ax6.scatter(sample['Time'], sample['Speed'], c=sample['hour'],
               cmap='twilight', s=2, alpha=0.5)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Speed')
    add_title_with_style(ax6, 'Activity Timeline', 'Sample colored by hour')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.savefig(output_path / 'temporal_patterns.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: temporal_patterns.png")


# ============================================================================
# ADVANCED ANALYSIS VISUALIZATIONS
# ============================================================================

def plot_behavioral_states(df: pd.DataFrame, output_path: Path,
                           figsize: Tuple[int, int] = (16, 10)) -> None:
    """Visualize behavioral state classification results."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # State colors
    state_colors = {
        'Resting': '#2ecc71',
        'Foraging': '#3498db',
        'Exploring': '#f39c12',
        'Traveling': '#e74c3c'
    }

    # 1. Spatial distribution by state
    ax1 = fig.add_subplot(gs[0, :2])
    sample = df.sample(n=min(30000, len(df)), random_state=42)
    for state, color in state_colors.items():
        mask = sample['state_label'] == state
        if mask.sum() > 0:
            ax1.scatter(sample.loc[mask, 'lon'], sample.loc[mask, 'lat'],
                       c=color, s=3, alpha=0.4, label=state)
    ax1.legend(loc='upper right', markerscale=3)
    add_title_with_style(ax1, 'Behavioral States - Spatial Distribution')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # 2. State proportions (pie chart)
    ax2 = fig.add_subplot(gs[0, 2])
    state_counts = df['state_label'].value_counts()
    colors = [state_colors.get(s, 'gray') for s in state_counts.index]
    wedges, texts, autotexts = ax2.pie(state_counts.values, labels=state_counts.index,
                                       colors=colors, autopct='%1.1f%%',
                                       explode=[0.02] * len(state_counts),
                                       shadow=True, startangle=90)
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    add_title_with_style(ax2, 'State Distribution')

    # 3. Speed distribution by state
    ax3 = fig.add_subplot(gs[1, 0])
    for state, color in state_colors.items():
        mask = df['state_label'] == state
        if mask.sum() > 0:
            ax3.hist(df.loc[mask, 'Speed'], bins=50, alpha=0.5,
                    color=color, label=state, density=True)
    ax3.set_xlabel('Speed')
    ax3.set_ylabel('Density')
    ax3.legend()
    add_title_with_style(ax3, 'Speed Distribution by State')

    # 4. State probability distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(df['state_probability'], bins=50, color='steelblue',
             edgecolor='white', alpha=0.7)
    ax4.axvline(df['state_probability'].mean(), color='red', linestyle='--',
               label=f'Mean: {df["state_probability"].mean():.2f}')
    ax4.set_xlabel('Classification Probability')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    add_title_with_style(ax4, 'Classification Confidence')

    # 5. State statistics table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    stats_by_state = df.groupby('state_label').agg({
        'Speed': ['mean', 'std'],
        'state_probability': 'mean',
        'lat': 'count'
    }).round(2)
    stats_by_state.columns = ['Speed Mean', 'Speed Std', 'Confidence', 'Count']

    table_data = [[state] + list(row) for state, row in stats_by_state.iterrows()]
    table = ax5.table(cellText=table_data,
                     colLabels=['State', 'Speed Mean', 'Speed Std', 'Confidence', 'Count'],
                     cellLoc='center',
                     loc='center',
                     colColours=['lightsteelblue'] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    add_title_with_style(ax5, 'State Statistics')

    plt.savefig(output_path / 'behavioral_states.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: behavioral_states.png")


def plot_movement_network(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                          output_path: Path, figsize: Tuple[int, int] = (14, 10)) -> None:
    """Visualize movement network analysis results."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Network nodes with type coloring
    ax = axes[0]

    node_colors = {
        'Hub': '#e74c3c',
        'Origin': '#3498db',
        'Destination': '#2ecc71',
        'Regular': '#bdc3c7'
    }
    node_sizes = {
        'Hub': 80,
        'Origin': 50,
        'Destination': 50,
        'Regular': 10
    }

    for node_type, color in node_colors.items():
        mask = nodes_df['node_type'] == node_type
        if mask.sum() > 0:
            size = node_sizes.get(node_type, 20)
            ax.scatter(nodes_df.loc[mask, 'lon'], nodes_df.loc[mask, 'lat'],
                      c=color, s=size, alpha=0.7, label=node_type, edgecolors='white')

    ax.legend(loc='upper right')
    add_title_with_style(ax, 'Movement Network Nodes', 'Hubs, Origins, Destinations')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 2. Node degree distribution
    ax = axes[1]
    if 'total_degree' in nodes_df.columns:
        ax.hist(nodes_df['total_degree'], bins=50, color='steelblue',
                edgecolor='white', alpha=0.7)
        ax.axvline(nodes_df['total_degree'].mean(), color='red', linestyle='--',
                  label=f'Mean: {nodes_df["total_degree"].mean():.1f}')
        ax.set_xlabel('Node Degree')
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')
        ax.legend()
        add_title_with_style(ax, 'Degree Distribution', 'Log scale')

    plt.tight_layout()
    plt.savefig(output_path / 'movement_network.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: movement_network.png")


def plot_flow_corridors(flow_df: pd.DataFrame, corridors_df: pd.DataFrame,
                        output_path: Path, figsize: Tuple[int, int] = (14, 10)) -> None:
    """Visualize movement flow and corridors."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. All flows
    ax = axes[0]

    # Draw flow lines with alpha based on count
    if len(flow_df) > 0:
        max_flow = flow_df['flow_count'].max()
        sample = flow_df.nlargest(500, 'flow_count')  # Top 500 flows

        for _, row in sample.iterrows():
            alpha = min(0.8, row['flow_count'] / max_flow + 0.1)
            linewidth = max(0.5, row['flow_count'] / max_flow * 3)
            ax.plot([row['origin_lon'], row['dest_lon']],
                   [row['origin_lat'], row['dest_lat']],
                   color='steelblue', alpha=alpha, linewidth=linewidth)

    add_title_with_style(ax, 'Movement Flows', 'Top 500 by volume')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 2. High-traffic corridors
    ax = axes[1]

    if len(corridors_df) > 0:
        max_flow = corridors_df['flow_count'].max()

        for _, row in corridors_df.head(100).iterrows():
            linewidth = max(1, row['flow_count'] / max_flow * 5)
            ax.plot([row['origin_lon'], row['dest_lon']],
                   [row['origin_lat'], row['dest_lat']],
                   color='red', alpha=0.6, linewidth=linewidth)

        # Add points at origins/destinations
        ax.scatter(corridors_df['origin_lon'], corridors_df['origin_lat'],
                  c='orange', s=20, alpha=0.5, zorder=5)
        ax.scatter(corridors_df['dest_lon'], corridors_df['dest_lat'],
                  c='green', s=20, alpha=0.5, zorder=5)

    add_title_with_style(ax, 'High-Traffic Corridors', 'Top 90th percentile')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()
    plt.savefig(output_path / 'flow_corridors.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: flow_corridors.png")


def plot_anomalies(df: pd.DataFrame, output_path: Path,
                   figsize: Tuple[int, int] = (14, 10)) -> None:
    """Visualize detected anomalies."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Spatial distribution of anomalies
    ax = axes[0]

    normal = df[~df['is_anomaly']]
    anomalies = df[df['is_anomaly']]

    # Plot normal points
    sample_normal = normal.sample(n=min(20000, len(normal)), random_state=42)
    ax.scatter(sample_normal['lon'], sample_normal['lat'], c='steelblue',
               s=1, alpha=0.2, label='Normal')

    # Plot anomalies
    sample_anomaly = anomalies.sample(n=min(5000, len(anomalies)), random_state=42)
    ax.scatter(sample_anomaly['lon'], sample_anomaly['lat'], c='red',
               s=10, alpha=0.6, label='Anomaly', marker='x')

    ax.legend()
    add_title_with_style(ax, 'Anomaly Spatial Distribution')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 2. Anomaly score distribution
    ax = axes[1]
    ax.hist(df['anomaly_score'], bins=100, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Decision boundary')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    add_title_with_style(ax, 'Anomaly Score Distribution')

    plt.tight_layout()
    plt.savefig(output_path / 'anomalies.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: anomalies.png")


def plot_ripleys_k(ripleys_data: dict, output_path: Path,
                   figsize: Tuple[int, int] = (14, 5)) -> None:
    """Visualize Ripley's K function results."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    distances = np.array(ripleys_data['distances'])
    k_values = np.array(ripleys_data['k_values'])
    l_values = np.array(ripleys_data['l_values'])
    expected_k = np.array(ripleys_data['expected_k'])

    # K function
    ax = axes[0]
    ax.plot(distances, k_values, 'b-', linewidth=2, label='Observed K(r)')
    ax.plot(distances, expected_k, 'r--', linewidth=2, label='Expected (CSR)')
    ax.fill_between(distances, k_values, expected_k,
                    where=(k_values > expected_k),
                    color='blue', alpha=0.3, label='Clustering')
    ax.fill_between(distances, k_values, expected_k,
                    where=(k_values < expected_k),
                    color='red', alpha=0.3, label='Dispersion')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('K(r)')
    ax.legend()
    add_title_with_style(ax, "Ripley's K Function")

    # L function (variance-stabilized)
    ax = axes[1]
    ax.plot(distances, l_values, 'b-', linewidth=2, label='L(r) - r')
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label='CSR expectation')
    ax.fill_between(distances, 0, l_values, where=(np.array(l_values) > 0),
                    color='blue', alpha=0.3)
    ax.fill_between(distances, 0, l_values, where=(np.array(l_values) < 0),
                    color='red', alpha=0.3)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('L(r) - r')
    ax.legend()
    add_title_with_style(ax, "L Function (Variance-Stabilized)")

    plt.tight_layout()
    plt.savefig(output_path / 'ripleys_k.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: ripleys_k.png")


# ============================================================================
# CREATIVE ANALYSIS VISUALIZATIONS
# ============================================================================

def plot_fractal_analysis(fractal_data: dict, fractal_ts: pd.DataFrame,
                          output_path: Path, figsize: Tuple[int, int] = (14, 10)) -> None:
    """Visualize fractal dimension analysis results."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Box-counting plot
    ax1 = fig.add_subplot(gs[0, 0])
    log_sizes = np.log(1 / np.array(fractal_data['box_sizes']))
    log_counts = np.log(fractal_data['box_counts'])

    ax1.scatter(log_sizes, log_counts, c='steelblue', s=60, zorder=5, edgecolors='white')

    # Fit line
    z = np.polyfit(log_sizes, log_counts, 1)
    p = np.poly1d(z)
    ax1.plot(log_sizes, p(log_sizes), 'r--', linewidth=2,
             label=f'D = {fractal_data["fractal_dimension"]:.3f}')

    ax1.set_xlabel('log(1/box size)')
    ax1.set_ylabel('log(box count)')
    ax1.legend(fontsize=12)
    add_title_with_style(ax1, 'Fractal Dimension', f'R² = {fractal_data["r_squared"]:.4f}')

    # 2. Interpretation gauge
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')

    fd = fractal_data['fractal_dimension']
    interp = fractal_data['interpretation']

    # Create gauge-like visualization
    gauge_text = f"""
    🔬 Fractal Analysis Results
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Fractal Dimension: {fd:.3f}
    
    Interpretation:
    {interp}
    
    Scale:
    < 1.2  → Linear (direct travel)
    1.2-1.5 → Moderate complexity
    1.5-1.8 → Complex exploration
    > 1.8  → Space-filling
    
    R² = {fractal_data['r_squared']:.4f}
    """
    ax2.text(0.1, 0.9, gauge_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))

    # 3. Time series of fractal dimension
    ax3 = fig.add_subplot(gs[1, :])
    if len(fractal_ts) > 0:
        ax3.plot(range(len(fractal_ts)), fractal_ts['fractal_dimension'],
                'o-', color='steelblue', linewidth=2, markersize=6)
        ax3.fill_between(range(len(fractal_ts)), 0, fractal_ts['fractal_dimension'],
                        alpha=0.3, color='steelblue')
        ax3.axhline(1.5, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
        ax3.set_xticks(range(0, len(fractal_ts), max(1, len(fractal_ts)//10)))
        ax3.set_xticklabels(fractal_ts['period'].values[::max(1, len(fractal_ts)//10)],
                           rotation=45, ha='right')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Fractal Dimension')
        ax3.legend()
    add_title_with_style(ax3, 'Movement Complexity Over Time')

    plt.savefig(output_path / 'fractal_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fractal_analysis.png")


def plot_entropy_analysis(entropy_data: dict, output_path: Path,
                          figsize: Tuple[int, int] = (14, 8)) -> None:
    """Visualize movement entropy analysis."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    metrics = ['spatial', 'temporal', 'transition']
    titles = ['Spatial Entropy', 'Temporal Entropy', 'Transition Entropy']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = fig.add_subplot(gs[0, i])

        data = entropy_data.get(metric, {})
        norm_entropy = data.get('normalized_entropy', 0)
        interpretation = data.get('interpretation', 'N/A')

        # Create bar with threshold indicator
        bars = ax.bar(['Value'], [norm_entropy], color=color, alpha=0.7, edgecolor='white', linewidth=2)
        ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Predictable/Unpredictable')

        ax.set_ylim(0, 1)
        ax.set_ylabel('Normalized Entropy')
        add_title_with_style(ax, title, interpretation)

        # Add value annotation
        ax.text(0, norm_entropy + 0.05, f'{norm_entropy:.3f}', ha='center',
                fontsize=14, fontweight='bold')

    plt.savefig(output_path / 'entropy_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: entropy_analysis.png")


def plot_sinuosity(sinuosity_df: pd.DataFrame, output_path: Path,
                   figsize: Tuple[int, int] = (14, 10)) -> None:
    """Visualize sinuosity (tortuosity) analysis."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Sinuosity distribution
    ax1 = fig.add_subplot(gs[0, 0])
    valid_sinuosity = sinuosity_df['sinuosity'].dropna()
    valid_sinuosity = valid_sinuosity[valid_sinuosity < valid_sinuosity.quantile(0.99)]  # Remove outliers

    ax1.hist(valid_sinuosity, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Straight line')
    ax1.axvline(valid_sinuosity.median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {valid_sinuosity.median():.2f}')
    ax1.set_xlabel('Sinuosity Index')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    add_title_with_style(ax1, 'Sinuosity Distribution')

    # 2. Movement type breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    if 'movement_type' in sinuosity_df.columns:
        type_counts = sinuosity_df['movement_type'].value_counts()
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        ax2.pie(type_counts.values, labels=type_counts.index, colors=colors[:len(type_counts)],
                autopct='%1.1f%%', explode=[0.02] * len(type_counts), shadow=True)
    add_title_with_style(ax2, 'Movement Type Distribution')

    # 3. Sinuosity vs Speed
    ax3 = fig.add_subplot(gs[1, 0])
    sample = sinuosity_df.dropna(subset=['sinuosity', 'mean_speed']).sample(
        n=min(1000, len(sinuosity_df)), random_state=42
    )
    scatter = ax3.scatter(sample['mean_speed'], sample['sinuosity'],
                         c=sample['actual_distance_km'], cmap='viridis',
                         alpha=0.6, s=30, edgecolors='white')
    plt.colorbar(scatter, ax=ax3, label='Distance (km)')
    ax3.set_xlabel('Mean Speed')
    ax3.set_ylabel('Sinuosity')
    add_title_with_style(ax3, 'Sinuosity vs Speed')

    # 4. Spatial distribution of sinuosity
    ax4 = fig.add_subplot(gs[1, 1])
    sample = sinuosity_df.dropna(subset=['sinuosity']).sample(
        n=min(1000, len(sinuosity_df)), random_state=42
    )
    scatter = ax4.scatter(sample['center_lon'], sample['center_lat'],
                         c=sample['sinuosity'].clip(upper=5),
                         cmap='RdYlGn_r', alpha=0.6, s=30)
    plt.colorbar(scatter, ax=ax4, label='Sinuosity')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    add_title_with_style(ax4, 'Spatial Sinuosity Pattern')

    plt.savefig(output_path / 'sinuosity_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: sinuosity_analysis.png")


def plot_revisitation(revisitation_df: pd.DataFrame, output_path: Path,
                      figsize: Tuple[int, int] = (14, 10)) -> None:
    """Visualize revisitation analysis results."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Spatial distribution by location type
    ax1 = fig.add_subplot(gs[0, :])

    type_colors = {
        'Core': '#e74c3c',
        'Frequent': '#f39c12',
        'Regular': '#3498db',
        'Transient': '#bdc3c7'
    }

    for loc_type in ['Transient', 'Regular', 'Frequent', 'Core']:  # Order matters for visibility
        mask = revisitation_df['location_type'] == loc_type
        if mask.sum() > 0:
            size = {'Core': 100, 'Frequent': 50, 'Regular': 20, 'Transient': 5}.get(loc_type, 10)
            ax1.scatter(revisitation_df.loc[mask, 'lon'],
                       revisitation_df.loc[mask, 'lat'],
                       c=type_colors[loc_type], s=size, alpha=0.6,
                       label=f'{loc_type} ({mask.sum():,})', edgecolors='white')

    ax1.legend(loc='upper right', markerscale=0.8)
    add_title_with_style(ax1, 'Location Types by Revisitation', 'Core = 20+ visits, Frequent = 10+')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # 2. Visit count distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(revisitation_df['distinct_visits'].clip(upper=50), bins=50,
             color='steelblue', edgecolor='white', alpha=0.7)
    ax2.set_xlabel('Distinct Visits')
    ax2.set_ylabel('Number of Locations')
    ax2.set_yscale('log')
    add_title_with_style(ax2, 'Visit Frequency Distribution', 'Log scale')

    # 3. Location type breakdown
    ax3 = fig.add_subplot(gs[1, 1])
    type_counts = revisitation_df['location_type'].value_counts()
    colors = [type_colors.get(t, 'gray') for t in type_counts.index]
    bars = ax3.barh(type_counts.index, type_counts.values, color=colors,
                   edgecolor='white', linewidth=2)
    ax3.set_xlabel('Number of Locations')
    add_title_with_style(ax3, 'Location Type Summary')

    # Add value labels
    for bar, count in zip(bars, type_counts.values):
        ax3.text(bar.get_width() + max(type_counts.values) * 0.02,
                bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=10)

    plt.savefig(output_path / 'revisitation_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: revisitation_analysis.png")


def plot_dynamic_home_range(hr_ts: pd.DataFrame, output_path: Path,
                            figsize: Tuple[int, int] = (14, 8)) -> None:
    """Visualize dynamic home range changes over time."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

    # 1. Home range area over time
    ax1 = axes[0]
    x = range(len(hr_ts))

    # Area with gradient fill
    ax1.fill_between(x, 0, hr_ts['area_km2'], alpha=0.4, color='steelblue')
    ax1.plot(x, hr_ts['area_km2'], 'o-', color='steelblue', linewidth=2, markersize=6)

    # Rolling average
    if len(hr_ts) > 3:
        rolling_avg = hr_ts['area_km2'].rolling(3, center=True).mean()
        ax1.plot(x, rolling_avg, 'r--', linewidth=2, alpha=0.7, label='3-period rolling avg')

    ax1.set_ylabel('Home Range Area (km²)')
    ax1.legend()
    add_title_with_style(ax1, 'Home Range Evolution Over Time')

    # X-axis labels
    if len(hr_ts) <= 20:
        ax1.set_xticks(x)
        ax1.set_xticklabels(hr_ts['period'], rotation=45, ha='right')
    else:
        step = len(hr_ts) // 10
        ax1.set_xticks(x[::step])
        ax1.set_xticklabels(hr_ts['period'].values[::step], rotation=45, ha='right')

    # 2. Centroid movement
    ax2 = axes[1]
    if 'centroid_lat' in hr_ts.columns and 'centroid_lon' in hr_ts.columns:
        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(hr_ts)))
        scatter = ax2.scatter(hr_ts['centroid_lon'], hr_ts['centroid_lat'],
                             c=range(len(hr_ts)), cmap='viridis', s=50,
                             edgecolors='white', linewidth=1)

        # Connect with lines
        ax2.plot(hr_ts['centroid_lon'], hr_ts['centroid_lat'], 'k-', alpha=0.3)

        # Mark start and end
        ax2.scatter(hr_ts['centroid_lon'].iloc[0], hr_ts['centroid_lat'].iloc[0],
                   c='green', s=150, marker='s', edgecolors='white', zorder=5, label='Start')
        ax2.scatter(hr_ts['centroid_lon'].iloc[-1], hr_ts['centroid_lat'].iloc[-1],
                   c='red', s=150, marker='s', edgecolors='white', zorder=5, label='End')

        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='Time', shrink=0.5)

    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    add_title_with_style(ax2, 'Centroid Drift Over Time')

    plt.tight_layout()
    plt.savefig(output_path / 'dynamic_home_range.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: dynamic_home_range.png")


def plot_significant_locations(sig_locs: pd.DataFrame, output_path: Path,
                               figsize: Tuple[int, int] = (14, 10)) -> None:
    """Visualize significant locations."""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Spatial distribution
    ax1 = fig.add_subplot(gs[0, :])

    type_colors = {
        'Primary Location': '#e74c3c',
        'Night Resting': '#9b59b6',
        'Day Activity': '#f39c12',
        'Regular Stop': '#3498db'
    }

    for loc_type, color in type_colors.items():
        mask = sig_locs['location_type'] == loc_type
        if mask.sum() > 0:
            size = sig_locs.loc[mask, 'total_duration_hours'] * 5
            ax1.scatter(sig_locs.loc[mask, 'lon'], sig_locs.loc[mask, 'lat'],
                       c=color, s=size.clip(20, 200), alpha=0.6,
                       label=f'{loc_type} ({mask.sum()})', edgecolors='white')

    ax1.legend(loc='upper right', markerscale=0.5)
    add_title_with_style(ax1, 'Significant Locations', 'Size = duration spent')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')

    # 2. Duration distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(sig_locs['total_duration_hours'].clip(upper=sig_locs['total_duration_hours'].quantile(0.95)),
             bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.set_xlabel('Total Duration (hours)')
    ax2.set_ylabel('Number of Locations')
    add_title_with_style(ax2, 'Time Spent Distribution')

    # 3. Peak hour distribution
    ax3 = fig.add_subplot(gs[1, 1])
    peak_hours = sig_locs['peak_hour'].value_counts().sort_index()
    colors = plt.cm.twilight(np.linspace(0, 1, 24))
    bars = ax3.bar(peak_hours.index, peak_hours.values,
                  color=[colors[h] for h in peak_hours.index],
                  edgecolor='white')
    ax3.set_xlabel('Peak Hour')
    ax3.set_ylabel('Number of Locations')
    ax3.set_xticks(range(0, 24, 3))
    add_title_with_style(ax3, 'Peak Activity Hours')

    plt.savefig(output_path / 'significant_locations.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: significant_locations.png")


def plot_day_night_patterns(day_night_df: pd.DataFrame, hourly_df: pd.DataFrame,
                            output_path: Path, figsize: Tuple[int, int] = (14, 8)) -> None:
    """Visualize day/night activity patterns."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Day vs Night comparison
    ax1 = axes[0]

    if len(day_night_df) >= 2:
        metrics = ['speed_mean', 'lat_spread']
        x = np.arange(len(metrics))
        width = 0.35

        day_data = day_night_df[day_night_df['period'] == 'Day']
        night_data = day_night_df[day_night_df['period'] == 'Night']

        if len(day_data) > 0 and len(night_data) > 0:
            day_values = [day_data['speed_mean'].values[0], day_data['lat_spread'].values[0]]
            night_values = [night_data['speed_mean'].values[0], night_data['lat_spread'].values[0]]

            ax1.bar(x - width/2, day_values, width, label='Day', color='#f39c12', alpha=0.8)
            ax1.bar(x + width/2, night_values, width, label='Night', color='#9b59b6', alpha=0.8)

            ax1.set_xticks(x)
            ax1.set_xticklabels(['Mean Speed', 'Spatial Spread'])
            ax1.legend()

    add_title_with_style(ax1, 'Day vs Night Activity')

    # 2. Hourly activity profile
    ax2 = axes[1]

    if len(hourly_df) > 0:
        hours = hourly_df['hour'] if 'hour' in hourly_df.columns else hourly_df.index
        activity = hourly_df['normalized_activity'] if 'normalized_activity' in hourly_df.columns else hourly_df['speed_mean'] / hourly_df['speed_mean'].max()

        # Day/night shading
        ax2.axvspan(6, 18, alpha=0.2, color='yellow', label='Day')
        ax2.axvspan(0, 6, alpha=0.2, color='purple')
        ax2.axvspan(18, 24, alpha=0.2, color='purple', label='Night')

        # Activity line
        ax2.fill_between(hours, 0, activity, alpha=0.4, color='steelblue')
        ax2.plot(hours, activity, 'o-', color='steelblue', linewidth=2, markersize=6)

        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Normalized Activity')
        ax2.set_xlim(0, 23)
        ax2.legend()

    add_title_with_style(ax2, 'Hourly Activity Profile')

    plt.tight_layout()
    plt.savefig(output_path / 'day_night_patterns.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: day_night_patterns.png")


def plot_activity_bursts(bursts_df: pd.DataFrame, output_path: Path,
                         figsize: Tuple[int, int] = (14, 8)) -> None:
    """Visualize activity bursts and lulls."""
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

    # 1. Timeline with bursts highlighted
    ax1 = axes[0]

    bursts_df = bursts_df.copy()
    # Handle various datetime formats
    try:
        bursts_df['Time'] = pd.to_datetime(bursts_df['Time'], format='ISO8601')
    except:
        bursts_df['Time'] = pd.to_datetime(bursts_df['Time'], format='mixed')

    # Sample for plotting
    sample = bursts_df.sample(n=min(10000, len(bursts_df)), random_state=42).sort_values('Time')

    # Plot all points
    ax1.scatter(sample['Time'], sample['Speed'], c='lightgray', s=1, alpha=0.3)

    # Highlight bursts
    bursts = sample[sample['burst_type'] == 'Activity Burst']
    lulls = sample[sample['burst_type'] == 'Activity Lull']

    ax1.scatter(bursts['Time'], bursts['Speed'], c='red', s=10, alpha=0.6, label='Burst')
    ax1.scatter(lulls['Time'], lulls['Speed'], c='blue', s=10, alpha=0.6, label='Lull')

    ax1.set_ylabel('Speed')
    ax1.legend()
    add_title_with_style(ax1, 'Activity Bursts and Lulls Over Time')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Z-score distribution
    ax2 = axes[1]
    ax2.hist(sample['activity_z'].dropna(), bins=100, color='steelblue',
             edgecolor='white', alpha=0.7)
    ax2.axvline(2, color='red', linestyle='--', linewidth=2, label='Burst threshold')
    ax2.axvline(-2, color='blue', linestyle='--', linewidth=2, label='Lull threshold')
    ax2.set_xlabel('Activity Z-Score')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    add_title_with_style(ax2, 'Activity Z-Score Distribution')

    plt.tight_layout()
    plt.savefig(output_path / 'activity_bursts.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: activity_bursts.png")


# ============================================================================
# COMPREHENSIVE DASHBOARDS
# ============================================================================

def create_standard_dashboard(output_path: Path, figsize: Tuple[int, int] = (20, 24)) -> None:
    """Create comprehensive dashboard for standard analysis."""

    # Check for summary file
    summary_file = output_path / 'analysis_summary.json'
    if not summary_file.exists():
        print("  No standard analysis summary found")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25)

    # Title
    fig.suptitle('CatDash Standard Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)

    # Summary metrics
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')

    # Safely format numeric values
    n_segments = summary.get('n_segments', 'N/A')
    n_segments_str = f"{n_segments:,}" if isinstance(n_segments, (int, float)) else str(n_segments)
    n_geohash = summary.get('n_geohash_cells', 0)
    n_geohash_str = f"{n_geohash:,}" if isinstance(n_geohash, (int, float)) else str(n_geohash)
    hr_area = summary.get('home_range_km2', 0)
    hr_area_str = f"{hr_area:.2f}" if isinstance(hr_area, (int, float)) else str(hr_area)
    morans = summary.get('morans_i', 0)
    morans_str = f"{morans:.4f}" if isinstance(morans, (int, float)) else str(morans)

    summary_text = f"""
    📊 Analysis Summary
    ━━━━━━━━━━━━━━━━━━
    
    Clusters: {summary.get('n_clusters', 'N/A')}
    Hotspots: {summary.get('n_hotspots', 'N/A')}
    Coldspots: {summary.get('n_coldspots', 'N/A')}
    Segments: {n_segments_str}
    
    Home Range: {hr_area_str} km²
    Geohash Cells: {n_geohash_str}
    
    Moran's I: {morans_str}
    Pattern: {summary.get('morans_interpretation', 'N/A')}
    """
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.5))

    # Embed existing plots
    plot_positions = [
        ('spatial_overview.png', gs[0, 1:]),
        ('hotspots_advanced.png', gs[1, :2]),
        ('temporal_patterns.png', gs[1, 2]),
        ('geohash_grid.png', gs[2, :2]),
        ('home_range.png', gs[2, 2]),
        ('speed_stats.png', gs[3, :]),
    ]

    for filename, grid_pos in plot_positions:
        filepath = output_path / filename
        if filepath.exists():
            ax = fig.add_subplot(grid_pos)
            img = plt.imread(filepath)
            ax.imshow(img)
            ax.axis('off')

    plt.savefig(output_path / 'standard_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: standard_dashboard.png")


def create_advanced_dashboard(output_path: Path, figsize: Tuple[int, int] = (20, 24)) -> None:
    """Create comprehensive dashboard for advanced analysis."""

    summary_file = output_path / 'advanced_analysis_summary.json'
    if not summary_file.exists():
        print("  No advanced analysis summary found")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25)

    fig.suptitle('CatDash Advanced Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)

    # Summary
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')

    # Safely format numeric values
    n_anomalies = summary.get('n_anomalies', 0)
    n_anomalies_str = f"{n_anomalies:,}" if isinstance(n_anomalies, (int, float)) else str(n_anomalies)
    max_overlap = summary.get('max_territory_overlap', 0)
    max_overlap_str = f"{max_overlap:.3f}" if isinstance(max_overlap, (int, float)) else str(max_overlap)

    summary_text = f"""
    🔬 Advanced Analysis
    ━━━━━━━━━━━━━━━━━
    
    Emerging Hotspots: {summary.get('emerging_hotspots', 'N/A')}
    Diminishing: {summary.get('diminishing_hotspots', 'N/A')}
    
    Network Hubs: {summary.get('n_network_hubs', 'N/A')}
    Anomalies: {n_anomalies_str}
    Corridors: {summary.get('n_corridors', 'N/A')}
    
    Territory Overlap: {max_overlap_str}
    Point Pattern: {summary.get('point_pattern', 'N/A')}
    """
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.5))

    # Behavioral states
    if 'behavioral_states' in summary:
        ax = fig.add_subplot(gs[0, 1])
        states = summary['behavioral_states']
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        ax.pie(list(states.values()), labels=list(states.keys()),
               colors=colors[:len(states)], autopct='%1.1f%%', startangle=90)
        ax.set_title('Behavioral States', fontweight='bold')

    # Embed existing plots
    plot_positions = [
        ('behavioral_states.png', gs[0, 2]),
        ('movement_network.png', gs[1, :2]),
        ('flow_corridors.png', gs[1, 2]),
        ('anomalies.png', gs[2, :2]),
        ('ripleys_k.png', gs[2, 2]),
    ]

    for filename, grid_pos in plot_positions:
        filepath = output_path / filename
        if filepath.exists():
            ax = fig.add_subplot(grid_pos)
            img = plt.imread(filepath)
            ax.imshow(img)
            ax.axis('off')

    plt.savefig(output_path / 'advanced_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: advanced_dashboard.png")


def create_creative_dashboard(output_path: Path, figsize: Tuple[int, int] = (20, 24)) -> None:
    """Create comprehensive dashboard for creative analysis."""

    summary_file = output_path / 'creative_analysis_summary.json'
    if not summary_file.exists():
        print("  No creative analysis summary found")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25)

    fig.suptitle('CatDash Creative Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)

    # Summary metrics
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    summary_text = f"""
    🎨 Creative Analysis
    ━━━━━━━━━━━━━━━━━
    
    Fractal Dimension: {summary.get('fractal_dimension', 0):.3f}
    ({summary.get('fractal_interpretation', 'N/A')})
    
    Persistence: {summary.get('mean_persistence', 0):.3f}
    ({summary.get('persistence_interpretation', 'N/A')})
    
    Spatial Entropy: {summary.get('spatial_entropy', 0):.3f}
    Temporal Entropy: {summary.get('temporal_entropy', 0):.3f}
    
    Activity Pattern: {summary.get('activity_pattern', 'N/A')}
    
    Core Locations: {summary.get('n_core_locations', 0)}
    Frequent Locations: {summary.get('n_frequent_locations', 0)}
    Significant Locations: {summary.get('n_significant_locations', 0)}
    
    Activity Bursts: {summary.get('n_activity_bursts', 0):,}
    Range Shifts: {summary.get('n_range_shifts', 0)}
    """
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='honeydew', alpha=0.5))

    # Embed existing plots
    plot_positions = [
        ('fractal_analysis.png', gs[0, 1:]),
        ('entropy_analysis.png', gs[1, :2]),
        ('sinuosity_analysis.png', gs[1, 2]),
        ('revisitation_analysis.png', gs[2, :2]),
        ('dynamic_home_range.png', gs[2, 2]),
        ('significant_locations.png', gs[3, :2]),
        ('day_night_patterns.png', gs[3, 2]),
    ]

    for filename, grid_pos in plot_positions:
        filepath = output_path / filename
        if filepath.exists():
            ax = fig.add_subplot(grid_pos)
            img = plt.imread(filepath)
            ax.imshow(img)
            ax.axis('off')

    plt.savefig(output_path / 'creative_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: creative_dashboard.png")


# ============================================================================
# MAIN VISUALIZATION RUNNER
# ============================================================================

def generate_standard_visualizations(output_path: Path, df: pd.DataFrame = None):
    """Generate all standard analysis visualizations."""
    print("\n📊 Generating Standard Analysis Visualizations...")

    # Spatial overview (needs data)
    if df is not None:
        try:
            plot_spatial_overview(df, output_path)
        except Exception as e:
            print(f"  ✗ Error in spatial_overview: {e}")

        try:
            plot_temporal_patterns(df, output_path)
        except Exception as e:
            print(f"  ✗ Error in temporal_patterns: {e}")

    # Hotspots
    hotspots_file = output_path / 'hotspots.csv'
    if hotspots_file.exists():
        try:
            hotspots_df = pd.read_csv(hotspots_file)
            plot_hotspots_advanced(hotspots_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in hotspots: {e}")

    # Create dashboard
    try:
        create_standard_dashboard(output_path)
    except Exception as e:
        print(f"  ✗ Error in standard_dashboard: {e}")


def generate_advanced_visualizations(output_path: Path, df: pd.DataFrame = None):
    """Generate all advanced analysis visualizations."""
    print("\n🔬 Generating Advanced Analysis Visualizations...")

    # Behavioral states
    states_file = output_path / 'behavioral_states.csv.gz'
    if states_file.exists():
        try:
            states_df = pd.read_csv(states_file, compression='gzip')
            plot_behavioral_states(states_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in behavioral_states: {e}")

    # Movement network
    nodes_file = output_path / 'network_nodes.csv'
    edges_file = output_path / 'network_edges.csv'
    if nodes_file.exists() and edges_file.exists():
        try:
            nodes_df = pd.read_csv(nodes_file)
            edges_df = pd.read_csv(edges_file)
            plot_movement_network(nodes_df, edges_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in movement_network: {e}")

    # Flow corridors
    flow_file = output_path / 'flow_matrix.csv'
    corridors_file = output_path / 'corridors.csv'
    if flow_file.exists() and corridors_file.exists():
        try:
            flow_df = pd.read_csv(flow_file)
            corridors_df = pd.read_csv(corridors_file)
            plot_flow_corridors(flow_df, corridors_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in flow_corridors: {e}")

    # Anomalies
    anomalies_file = output_path / 'anomalies.csv.gz'
    if anomalies_file.exists():
        try:
            anomalies_df = pd.read_csv(anomalies_file, compression='gzip')
            plot_anomalies(anomalies_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in anomalies: {e}")

    # Ripley's K
    ripleys_file = output_path / 'ripleys_k.json'
    if ripleys_file.exists():
        try:
            with open(ripleys_file) as f:
                ripleys_data = json.load(f)
            plot_ripleys_k(ripleys_data, output_path)
        except Exception as e:
            print(f"  ✗ Error in ripleys_k: {e}")

    # Create dashboard
    try:
        create_advanced_dashboard(output_path)
    except Exception as e:
        print(f"  ✗ Error in advanced_dashboard: {e}")


def generate_creative_visualizations(output_path: Path, df: pd.DataFrame = None):
    """Generate all creative analysis visualizations."""
    print("\n🎨 Generating Creative Analysis Visualizations...")

    # Fractal analysis
    fractal_file = output_path / 'fractal_dimension.json'
    fractal_ts_file = output_path / 'fractal_time_series.csv'
    if fractal_file.exists():
        try:
            with open(fractal_file) as f:
                fractal_data = json.load(f)
            fractal_ts = pd.read_csv(fractal_ts_file) if fractal_ts_file.exists() else pd.DataFrame()
            plot_fractal_analysis(fractal_data, fractal_ts, output_path)
        except Exception as e:
            print(f"  ✗ Error in fractal_analysis: {e}")

    # Entropy analysis
    entropy_file = output_path / 'movement_entropy.json'
    if entropy_file.exists():
        try:
            with open(entropy_file) as f:
                entropy_data = json.load(f)
            plot_entropy_analysis(entropy_data, output_path)
        except Exception as e:
            print(f"  ✗ Error in entropy_analysis: {e}")

    # Sinuosity
    sinuosity_file = output_path / 'sinuosity_analysis.csv'
    if sinuosity_file.exists():
        try:
            sinuosity_df = pd.read_csv(sinuosity_file)
            plot_sinuosity(sinuosity_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in sinuosity: {e}")

    # Revisitation
    revisit_file = output_path / 'revisitation_analysis.csv'
    if revisit_file.exists():
        try:
            revisit_df = pd.read_csv(revisit_file)
            plot_revisitation(revisit_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in revisitation: {e}")

    # Dynamic home range
    hr_file = output_path / 'home_range_time_series.csv'
    if hr_file.exists():
        try:
            hr_df = pd.read_csv(hr_file)
            plot_dynamic_home_range(hr_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in dynamic_home_range: {e}")

    # Significant locations
    sig_file = output_path / 'significant_locations.csv'
    if sig_file.exists():
        try:
            sig_df = pd.read_csv(sig_file)
            plot_significant_locations(sig_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in significant_locations: {e}")

    # Day/night patterns
    day_night_file = output_path / 'day_night_patterns.csv'
    hourly_file = output_path / 'hourly_activity_profile.csv'
    if day_night_file.exists() and hourly_file.exists():
        try:
            day_night_df = pd.read_csv(day_night_file)
            hourly_df = pd.read_csv(hourly_file)
            plot_day_night_patterns(day_night_df, hourly_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in day_night_patterns: {e}")

    # Activity bursts
    bursts_file = output_path / 'activity_bursts.csv'
    if bursts_file.exists():
        try:
            bursts_df = pd.read_csv(bursts_file)
            if len(bursts_df) > 0:
                plot_activity_bursts(bursts_df, output_path)
        except Exception as e:
            print(f"  ✗ Error in activity_bursts: {e}")

    # Create dashboard
    try:
        create_creative_dashboard(output_path)
    except Exception as e:
        print(f"  ✗ Error in creative_dashboard: {e}")


def generate_all_visualizations(results_dir: str, data_file: str = None):
    """Generate all visualizations for all analysis types."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory {results_dir} not found")
        return

    print("=" * 60)
    print("CatDash Visualization Suite")
    print("=" * 60)

    # Load data if provided
    df = None
    if data_file and Path(data_file).exists():
        print(f"\nLoading data from {data_file}...")
        df = pd.read_csv(data_file, sep='\t', compression='gzip' if data_file.endswith('.gz') else None)
        print(f"Loaded {len(df):,} records")

    # Or try to load enriched data
    enriched_file = results_path / 'standard' / 'enriched_data.csv.gz'
    if df is None and enriched_file.exists():
        print(f"\nLoading enriched data...")
        df = pd.read_csv(enriched_file, compression='gzip')
        print(f"Loaded {len(df):,} records")

    # Generate visualizations for each analysis type
    standard_path = results_path / 'standard'
    if standard_path.exists():
        generate_standard_visualizations(standard_path, df)

    advanced_path = results_path / 'advanced'
    if advanced_path.exists():
        generate_advanced_visualizations(advanced_path, df)

    creative_path = results_path / 'creative'
    if creative_path.exists():
        generate_creative_visualizations(creative_path, df)

    print("\n" + "=" * 60)
    print("✅ Visualization generation complete!")
    print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate CatDash Analysis Visualizations')
    parser.add_argument('-i', '--input', type=str, default='output/results',
                        help='Results directory (containing standard/advanced/creative subdirs)')
    parser.add_argument('-d', '--data', type=str, default=None,
                        help='Optional: raw data file for additional plots')
    parser.add_argument('-t', '--type', type=str, default='all',
                        choices=['all', 'standard', 'advanced', 'creative'],
                        help='Which visualization set to generate')

    args = parser.parse_args()

    if args.type == 'all':
        generate_all_visualizations(args.input, args.data)
    else:
        results_path = Path(args.input) / args.type
        if not results_path.exists():
            print(f"Directory {results_path} not found")
            return

        # Load data
        df = None
        if args.data:
            df = pd.read_csv(args.data, sep='\t',
                           compression='gzip' if args.data.endswith('.gz') else None)

        if args.type == 'standard':
            generate_standard_visualizations(results_path, df)
        elif args.type == 'advanced':
            generate_advanced_visualizations(results_path, df)
        elif args.type == 'creative':
            generate_creative_visualizations(results_path, df)


if __name__ == '__main__':
    main()
