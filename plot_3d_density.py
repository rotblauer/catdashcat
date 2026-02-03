#!/usr/bin/env python3
"""
3D Density Plot for Minnesota GPS Data
Creates a stunning 3D surface visualization of point density.
With state boundary overlay.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# Use non-interactive backend
plt.switch_backend('Agg')


def load_minnesota_boundary(shapefile_path: str = 'output/cb_2020_us_county_500k.shp'):
    """
    Load Minnesota state boundary from county shapefile.
    Returns list of boundary polygons as (lon, lat) arrays.
    """
    try:
        import shapefile
    except ImportError:
        print("  Installing pyshp for shapefile support...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'pyshp', '-q'])
        import shapefile

    sf = shapefile.Reader(shapefile_path)

    mn_boundaries = []

    # Minnesota FIPS code is '27'
    for shaperec in sf.iterShapeRecords():
        record = shaperec.record
        # STATEFP is typically the first field - check for Minnesota
        state_fip = str(record[0]) if record else ''

        if state_fip == '27':  # Minnesota
            shape = shaperec.shape
            # Handle multi-part polygons
            if shape.shapeType in [5, 15]:  # Polygon or PolygonZ
                parts = list(shape.parts) + [len(shape.points)]
                for i in range(len(parts) - 1):
                    pts = shape.points[parts[i]:parts[i+1]]
                    coords = np.array(pts)
                    mn_boundaries.append(coords)

    return mn_boundaries


def get_state_outline(boundaries):
    """
    Get the outer boundary of the state from county boundaries.
    Returns simplified outline coordinates.
    """
    if not boundaries:
        return None

    # Combine all boundary points
    all_points = np.vstack(boundaries)

    # Get convex hull or use all boundaries
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(all_points)
        outline = all_points[hull.vertices]
        # Close the polygon
        outline = np.vstack([outline, outline[0]])
        return outline
    except:
        return None


def create_3d_density_plot(data_file: str, output_file: str,
                           lat_bounds: tuple = None, lon_bounds: tuple = None,
                           resolution: int = 150, sigma: float = 2.0,
                           show_boundary: bool = True):
    """
    Create a 3D density surface plot.

    Args:
        data_file: Path to input TSV file
        output_file: Path for output PNG
        lat_bounds: (min_lat, max_lat) or None for auto
        lon_bounds: (min_lon, max_lon) or None for auto
        resolution: Grid resolution for density calculation
        sigma: Gaussian smoothing parameter
        show_boundary: Whether to overlay state boundary
    """
    print("Loading data...")
    df = pd.read_csv(data_file, sep='\t',
                     compression='gzip' if data_file.endswith('.gz') else None)
    print(f"Loaded {len(df):,} records")

    # Apply bounds filter
    if lat_bounds and lon_bounds:
        mask = ((df['lat'] >= lat_bounds[0]) & (df['lat'] <= lat_bounds[1]) &
                (df['lon'] >= lon_bounds[0]) & (df['lon'] <= lon_bounds[1]))
        df = df[mask].copy()
        print(f"Filtered to {len(df):,} records in bounds")

    if len(df) == 0:
        print("No data in specified bounds!")
        return

    # Create 2D histogram for density
    # Use full bounds for histogram extent so boundaries align correctly
    print("Computing density...")
    if lat_bounds and lon_bounds:
        hist_range = [[lon_bounds[0], lon_bounds[1]], [lat_bounds[0], lat_bounds[1]]]
    else:
        hist_range = None

    hist, xedges, yedges = np.histogram2d(
        df['lon'].values,
        df['lat'].values,
        bins=resolution,
        range=hist_range
    )

    # Smooth the density
    hist_smooth = gaussian_filter(hist, sigma=sigma)

    # Create meshgrid for surface
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)

    # Log transform for better visibility
    Z = np.log1p(hist_smooth.T)
    z_max = Z.max()

    # Create figure
    print("Creating 3D visualization...")
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Custom colormap (plasma-like)
    colors = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786',
              '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']
    cmap = LinearSegmentedColormap.from_list('plasma_custom', colors)

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                           linewidth=0, antialiased=True,
                           alpha=0.95, rstride=1, cstride=1,
                           shade=True)

    # Add state boundary overlay
    if show_boundary:
        print("Loading state boundary...")
        try:
            boundaries = load_minnesota_boundary()
            if boundaries:
                print(f"  Found {len(boundaries)} county boundaries")

                # Draw each county boundary prominently
                for boundary in boundaries:
                    lons = boundary[:, 0]
                    lats = boundary[:, 1]

                    # Filter to bounds
                    if lat_bounds and lon_bounds:
                        mask = ((lats >= lat_bounds[0] - 0.5) & (lats <= lat_bounds[1] + 0.5) &
                               (lons >= lon_bounds[0] - 0.5) & (lons <= lon_bounds[1] + 0.5))
                        if not mask.any():
                            continue

                    # Draw boundary at base with black outline for contrast
                    z_base = np.zeros_like(lons)
                    ax.plot(lons, lats, z_base, color='black', linewidth=2.5, alpha=0.9, zorder=10)
                    ax.plot(lons, lats, z_base, color='white', linewidth=1.5, alpha=1.0, zorder=11)

                    # Draw boundary floating above the surface for maximum visibility
                    z_top = np.full_like(lons, z_max * 1.05)
                    ax.plot(lons, lats, z_top, color='black', linewidth=3.0, alpha=0.9, zorder=10)
                    ax.plot(lons, lats, z_top, color='#00FF00', linewidth=2.0, alpha=1.0, zorder=11)  # Bright green

                print("  ✓ State boundary added")
        except Exception as e:
            print(f"  Could not load boundary: {e}")

    # Labels and title
    ax.set_xlabel('\nLongitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('\nLatitude', fontsize=12, fontweight='bold')
    ax.set_zlabel('\nLog Density', fontsize=12, fontweight='bold')

    title = f'3D Activity Density Map\n{len(df):,} GPS Points'
    if lat_bounds:
        title += f'\nMinnesota Region'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.08)
    cbar.set_label('Log(Density + 1)', fontsize=11, fontweight='bold')

    # Viewing angle
    ax.view_init(elev=30, azim=-50)

    # Style the panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"✓ Saved: {output_file}")


def create_multiple_views(data_file: str, output_dir: str,
                          lat_bounds: tuple = None, lon_bounds: tuple = None,
                          show_boundary: bool = True):
    """Create 3D density from multiple viewing angles."""

    print("Loading data...")
    df = pd.read_csv(data_file, sep='\t',
                     compression='gzip' if data_file.endswith('.gz') else None)

    if lat_bounds and lon_bounds:
        mask = ((df['lat'] >= lat_bounds[0]) & (df['lat'] <= lat_bounds[1]) &
                (df['lon'] >= lon_bounds[0]) & (df['lon'] <= lon_bounds[1]))
        df = df[mask].copy()

    print(f"Using {len(df):,} points")

    # Compute density once - use full bounds for proper alignment
    resolution = 150
    if lat_bounds and lon_bounds:
        hist_range = [[lon_bounds[0], lon_bounds[1]], [lat_bounds[0], lat_bounds[1]]]
    else:
        hist_range = None

    hist, xedges, yedges = np.histogram2d(df['lon'].values, df['lat'].values,
                                          bins=resolution, range=hist_range)
    hist_smooth = gaussian_filter(hist, sigma=2.0)

    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers)
    Z = np.log1p(hist_smooth.T)
    z_max = Z.max()

    colors = ['#0d0887', '#46039f', '#7201a8', '#9c179e', '#bd3786',
              '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']
    cmap = LinearSegmentedColormap.from_list('plasma_custom', colors)

    # Load boundary once
    boundaries = None
    if show_boundary:
        try:
            print("Loading state boundary...")
            boundaries = load_minnesota_boundary()
            print(f"  Found {len(boundaries)} county boundaries")
        except Exception as e:
            print(f"  Could not load boundary: {e}")

    # Multiple views
    views = [
        (30, -50, 'view_front'),
        (30, -140, 'view_back'),
        (60, -50, 'view_top'),
        (15, -50, 'view_side'),
    ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for elev, azim, name in views:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0,
                               antialiased=True, alpha=0.95)

        # Add boundary overlay
        if boundaries:
            for boundary in boundaries:
                lons = boundary[:, 0]
                lats = boundary[:, 1]

                if lat_bounds and lon_bounds:
                    mask = ((lats >= lat_bounds[0] - 0.5) & (lats <= lat_bounds[1] + 0.5) &
                           (lons >= lon_bounds[0] - 0.5) & (lons <= lon_bounds[1] + 0.5))
                    if not mask.any():
                        continue

                # Base boundary
                z_base = np.zeros_like(lons)
                ax.plot(lons, lats, z_base, color='black', linewidth=2.5, alpha=0.9, zorder=10)
                ax.plot(lons, lats, z_base, color='white', linewidth=1.5, alpha=1.0, zorder=11)

                # Top boundary (floating above surface)
                z_top = np.full_like(lons, z_max * 1.05)
                ax.plot(lons, lats, z_top, color='black', linewidth=3.0, alpha=0.9, zorder=10)
                ax.plot(lons, lats, z_top, color='#00FF00', linewidth=2.0, alpha=1.0, zorder=11)

        ax.set_xlabel('\nLongitude', fontsize=11)
        ax.set_ylabel('\nLatitude', fontsize=11)
        ax.set_zlabel('\nLog Density', fontsize=11)
        ax.set_title(f'3D Density - Minnesota\n{len(df):,} points | elev={elev}°, azim={azim}°',
                     fontsize=14, fontweight='bold')

        ax.view_init(elev=elev, azim=azim)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.08)

        outfile = output_path / f'minnesota_3d_{name}.png'
        plt.savefig(outfile, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved: {outfile}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create 3D Density Plot')
    parser.add_argument('-i', '--input', default='output/raw.tsv.gz', help='Input data file')
    parser.add_argument('-o', '--output', default='output/results/minnesota_3d_density.png',
                        help='Output image file')
    parser.add_argument('--mn', action='store_true', help='Filter to Minnesota bounds')
    parser.add_argument('--multi', action='store_true', help='Generate multiple views')
    parser.add_argument('--no-boundary', action='store_true', help='Skip state boundary overlay')

    args = parser.parse_args()

    # Minnesota bounds
    MN_BOUNDS = ((43.5, 49.4), (-97.2, -89.5))  # (lat_bounds, lon_bounds)

    lat_bounds = MN_BOUNDS[0] if args.mn else None
    lon_bounds = MN_BOUNDS[1] if args.mn else None
    show_boundary = not args.no_boundary

    if args.multi:
        create_multiple_views(args.input, 'output/results', lat_bounds, lon_bounds, show_boundary)
    else:
        create_3d_density_plot(args.input, args.output, lat_bounds, lon_bounds,
                               show_boundary=show_boundary)
