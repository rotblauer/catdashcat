#!/usr/bin/env python3
"""Create a visually stunning 3D density map of cat GPS data across the entire US.

Memory-efficient version using chunked reading and incremental histogram binning.
Suitable for datasets with millions of points.

Example usage:
    .venv/bin/python us_density_map.py --resolution 2000 -i output/raw.tsv.gz --sigma 0.8 --power 3.5 -o output/results/us_density_3d_perspective.pdf
    python us_density_map.py -i output/raw.tsv.gz --resolution 600 --sigma 1.0 --power 3.0
    python us_density_map.py --sigma 0.8 --power 3.5  # More peaky: less smoothing + stronger exponent
    python us_density_map.py --chunk-size 250000  # Lower chunk size for less memory
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import shapefile

plt.switch_backend('Agg')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# --- Configuration (can be overridden via CLI) ---
DEFAULT_RESOLUTION = 500
DEFAULT_SIGMA = 1.2
DEFAULT_POWER = 2.5
DEFAULT_CHUNK_SIZE = 500_000

# Continental US bounds
US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -125.0, -66.5


def build_histogram_chunked(input_file, resolution, chunk_size):
    """
    Build a 2D histogram incrementally from chunked file reads.
    Only loads lat/lon columns to minimize memory usage.
    """
    # Preallocate histogram with fixed bin edges
    lon_edges = np.linspace(US_LON_MIN, US_LON_MAX, resolution + 1, dtype=np.float32)
    lat_edges = np.linspace(US_LAT_MIN, US_LAT_MAX, resolution + 1, dtype=np.float32)
    hist = np.zeros((resolution, resolution), dtype=np.float32)

    total_points = 0
    chunks_processed = 0

    print(f"Reading data in chunks of {chunk_size:,}...")

    # Stream through file, only reading lat/lon columns
    for chunk in pd.read_csv(
        input_file,
        sep='\t',
        compression='gzip',
        usecols=['lat', 'lon'],
        dtype={'lat': 'str', 'lon': 'str'},  # Read as string first for robust coercion
        chunksize=chunk_size,
        on_bad_lines='skip'
    ):
        # Convert to numeric, coercing errors to NaN
        chunk['lat'] = pd.to_numeric(chunk['lat'], errors='coerce')
        chunk['lon'] = pd.to_numeric(chunk['lon'], errors='coerce')

        # Filter to continental US bounds and drop NaN
        mask = (
            chunk['lat'].between(US_LAT_MIN, US_LAT_MAX) &
            chunk['lon'].between(US_LON_MIN, US_LON_MAX)
        )
        filtered = chunk.loc[mask].dropna()

        if filtered.empty:
            chunks_processed += 1
            continue

        # Accumulate into histogram
        h, _, _ = np.histogram2d(
            filtered['lon'].values,
            filtered['lat'].values,
            bins=[lon_edges, lat_edges]
        )
        hist += h.astype(np.float32)
        total_points += len(filtered)
        chunks_processed += 1

        if chunks_processed % 10 == 0:
            print(f"  Processed {chunks_processed} chunks, {total_points:,} points so far...")

    print(f"✓ Finished: {total_points:,} points in continental US from {chunks_processed} chunks")
    return hist, lon_edges, lat_edges, total_points


def main():
    parser = argparse.ArgumentParser(description='Create 3D density map of GPS data')
    parser.add_argument('-i', '--input', default='output/raw.tsv.gz', help='Input TSV file')
    parser.add_argument('-o', '--output', default='output/results/us_density_3d_perspective.pdf',
                        help='Output PDF file')
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION,
                        help=f'Histogram resolution (default: {DEFAULT_RESOLUTION})')
    parser.add_argument('--sigma', type=float, default=DEFAULT_SIGMA,
                        help=f'Gaussian smoothing sigma (default: {DEFAULT_SIGMA}). '
                             f'Lower values (0.5-1.0) create sharper, more defined peaks. '
                             f'Higher values (2.0+) produce smoother, more blended terrain.')
    parser.add_argument('--power', type=float, default=DEFAULT_POWER,
                        help=f'Power exponent for peak emphasis (default: {DEFAULT_POWER}). '
                             f'Higher values (3.0-4.0) make peaks taller and valleys deeper, '
                             f'creating dramatic contrast. Lower values (1.5-2.0) flatten the '
                             f'terrain for a more subtle visualization.')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Rows per chunk (default: {DEFAULT_CHUNK_SIZE:,})')
    args = parser.parse_args()

    # Build histogram incrementally
    hist, lon_edges, lat_edges, total_points = build_histogram_chunked(
        args.input, args.resolution, args.chunk_size
    )
    # Light smoothing to reduce noise but preserve peak sharpness (in-place for memory)
    print(f"Applying gaussian smoothing (sigma={args.sigma})...")
    hist_smooth = gaussian_filter(hist, sigma=args.sigma, output=hist)  # reuse buffer
    hist_smooth = hist  # alias after in-place operation

    # Compute bin centers (reuse edges, no extra copy)
    x_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    y_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    X, Y = np.meshgrid(x_centers, y_centers, copy=False)

    # Use power scaling to make peaks more prominent
    Z_raw = np.log1p(hist_smooth.T)
    Z = np.power(Z_raw, args.power)

    # Create a masked array to make only true zero density areas transparent
    Z_masked = np.ma.masked_where(hist_smooth.T == 0, Z)

    print(f"X (lon) range: {X.min():.4f} to {X.max():.4f}")
    print(f"Y (lat) range: {Y.min():.4f} to {Y.max():.4f}")
    print(f"Z range: {Z.min():.4f} to {Z.max():.4f}")
    print(f"Masked cells: {Z_masked.mask.sum():,} / {Z.size:,} ({100*Z_masked.mask.sum()/Z.size:.1f}%)")

    # Load US state boundaries
    print("Loading state boundaries...")
    sf = shapefile.Reader('output/cb_2020_us_county_500k.shp')

    # Get unique states (use state FIPS codes for state outlines)
    state_boundaries = {}
    for shaperec in sf.iterShapeRecords():
        record = shaperec.record
        state_fip = str(record[0]) if record else ''

        # Skip non-continental US (Alaska=02, Hawaii=15, Puerto Rico=72, etc.)
        if state_fip in ['02', '15', '72', '78', '60', '66', '69']:
            continue

        shape = shaperec.shape
        if shape.shapeType in [5, 15]:
            parts = list(shape.parts) + [len(shape.points)]
            for i in range(len(parts) - 1):
                pts = shape.points[parts[i]:parts[i+1]]
                coords = np.array(pts)
                if state_fip not in state_boundaries:
                    state_boundaries[state_fip] = []
                state_boundaries[state_fip].append(coords)

    # Flatten all boundaries
    all_boundaries = []
    for state, boundaries in state_boundaries.items():
        all_boundaries.extend(boundaries)

    print(f"Loaded {len(all_boundaries)} boundary polygons from {len(state_boundaries)} states")

    # Create publication-quality perspective view
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Use a custom colormap for better visual appeal
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#1a1a2e', '#16213e', '#0f3460', '#e94560', '#ff6b6b', '#feca57', '#fff9db']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('density', colors, N=n_bins)

    # Plot the density surface
    surf = ax.plot_surface(X, Y, Z_masked, cmap=cmap, alpha=0.95,
                            linewidth=0, antialiased=True,
                            rstride=1, cstride=1)

    # Draw state/county boundaries
    boundary_z = -0.3
    for boundary in all_boundaries:
        lons = boundary[:, 0]
        lats = boundary[:, 1]
        if lons.max() >= US_LON_MIN and lons.min() <= US_LON_MAX and \
           lats.max() >= US_LAT_MIN and lats.min() <= US_LAT_MAX:
            ax.plot(lons, lats, zs=boundary_z, zdir='z', color='#555555', linewidth=0.4, alpha=0.6)

    # Set perspective view
    ax.view_init(elev=60, azim=-100)

    # Remove all axes for clean look
    ax.set_axis_off()

    # Adjust aspect ratio
    ax.set_box_aspect([1.5, 1, 0.9])  # Taller Z for more dramatic peaks

    # Add elegant title
    ax.set_title('Spatial Distribution of Cat GPS Activity\nContinental United States',
                 fontsize=20, fontweight='bold', pad=30, color='#333333')

    # Add colorbar with better styling
    cbar = fig.colorbar(surf, ax=ax, shrink=0.35, aspect=20, pad=0.02,
                        orientation='vertical', location='right')
    cbar.set_label('Relative Activity Density', fontsize=12, color='#333333')
    cbar.ax.tick_params(labelsize=10, colors='#333333')
    cbar.outline.set_edgecolor('#cccccc')
    cbar.outline.set_linewidth(0.5)

    # Add data attribution
    fig.text(0.02, 0.02, f'n = {total_points:,} GPS observations',
             fontsize=10, color='#666666', style='italic')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: {args.output}")

    print("\n🎉 Done! Created publication-quality density map.")


if __name__ == '__main__':
    main()

