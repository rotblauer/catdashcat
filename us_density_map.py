#!/usr/bin/env python3
"""Create a visually stunning 3D density map of cat GPS data across the entire US."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import shapefile

plt.switch_backend('Agg')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

print("Loading data...")
df = pd.read_csv('output/raw.tsv.gz', sep='\t', compression='gzip', low_memory=False, on_bad_lines='skip')

# Convert lat/lon to numeric, coercing errors to NaN
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

# Filter to continental US bounds
US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -125.0, -66.5

us_mask = (df['lat'] >= US_LAT_MIN) & (df['lat'] <= US_LAT_MAX) & \
          (df['lon'] >= US_LON_MIN) & (df['lon'] <= US_LON_MAX)
df_us = df[us_mask]
print(f"Using {len(df_us):,} points in continental US")

# Create 2D histogram
resolution = 500  # Higher resolution for more precise peaks
hist_range = [[US_LON_MIN, US_LON_MAX], [US_LAT_MIN, US_LAT_MAX]]
hist, xedges, yedges = np.histogram2d(df_us['lon'].values, df_us['lat'].values,
                                       bins=resolution, range=hist_range)
# Light smoothing to reduce noise but preserve peak sharpness
hist_smooth = gaussian_filter(hist, sigma=1.2)

x_centers = (xedges[:-1] + xedges[1:]) / 2
y_centers = (yedges[:-1] + yedges[1:]) / 2
X, Y = np.meshgrid(x_centers, y_centers)

# Use power scaling to make peaks more prominent
Z_raw = np.log1p(hist_smooth.T)
Z = np.power(Z_raw, 2.5)  # Even stronger exaggeration of peaks

# Create a masked array to make only true zero density areas transparent
# Mask based on raw histogram (before log transform) to only hide cells with no data
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
fig.text(0.02, 0.02, f'n = {len(df_us):,} GPS observations',
         fontsize=10, color='#666666', style='italic')

plt.tight_layout()
plt.savefig('output/results/us_density_3d_perspective.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ Saved: output/results/us_density_3d_perspective.pdf")

print("\n🎉 Done! Created publication-quality density map.")
