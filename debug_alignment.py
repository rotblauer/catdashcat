#!/usr/bin/env python3
"""Debug script to verify coordinate alignment between density surface and boundaries."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import shapefile

plt.switch_backend('Agg')

print("Loading data...")
df = pd.read_csv('output/raw.tsv.gz', sep='\t', compression='gzip')
mn_mask = (df['lat'] >= 43.5) & (df['lat'] <= 49.4) & (df['lon'] >= -97.2) & (df['lon'] <= -89.5)
df_mn = df[mn_mask]
print(f"Using {len(df_mn):,} points")

# Create 2D histogram - use full MN bounds for proper alignment
resolution = 150
MN_LAT_MIN, MN_LAT_MAX = 43.5, 49.4
MN_LON_MIN, MN_LON_MAX = -97.2, -89.5
hist_range = [[MN_LON_MIN, MN_LON_MAX], [MN_LAT_MIN, MN_LAT_MAX]]
hist, xedges, yedges = np.histogram2d(df_mn['lon'].values, df_mn['lat'].values,
                                       bins=resolution, range=hist_range)
hist_smooth = gaussian_filter(hist, sigma=2.0)

x_centers = (xedges[:-1] + xedges[1:]) / 2
y_centers = (yedges[:-1] + yedges[1:]) / 2
X, Y = np.meshgrid(x_centers, y_centers)
Z = np.log1p(hist_smooth.T)

print(f"X (lon) range: {X.min():.4f} to {X.max():.4f}")
print(f"Y (lat) range: {Y.min():.4f} to {Y.max():.4f}")
print(f"Z max location: X={X.flat[Z.argmax()]:.4f}, Y={Y.flat[Z.argmax()]:.4f}")

# Load MN boundaries
sf = shapefile.Reader('output/cb_2020_us_county_500k.shp')
mn_boundaries = []
for shaperec in sf.iterShapeRecords():
    record = shaperec.record
    state_fip = str(record[0]) if record else ''
    if state_fip == '27':
        shape = shaperec.shape
        if shape.shapeType in [5, 15]:
            parts = list(shape.parts) + [len(shape.points)]
            for i in range(len(parts) - 1):
                pts = shape.points[parts[i]:parts[i+1]]
                coords = np.array(pts)
                mn_boundaries.append(coords)

# Create 2D top-down view to check alignment
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Left: 2D density heatmap with boundaries
ax = axes[0]
im = ax.pcolormesh(X, Y, Z, cmap='plasma', shading='auto')
plt.colorbar(im, ax=ax, label='Log Density')

# Overlay boundaries
for boundary in mn_boundaries:
    lons = boundary[:, 0]
    lats = boundary[:, 1]
    ax.plot(lons, lats, 'g-', linewidth=1.0, alpha=0.8)

# Mark peak density
peak_idx = np.unravel_index(Z.argmax(), Z.shape)
peak_lon = X[peak_idx]
peak_lat = Y[peak_idx]
ax.plot(peak_lon, peak_lat, 'r*', markersize=20, label=f'Peak: ({peak_lon:.2f}, {peak_lat:.2f})')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('2D View: Density + County Boundaries\n(Green = County lines)')
ax.legend()
ax.set_aspect('equal')

# Right: Scatter of raw GPS points with boundaries
ax = axes[1]
sample = df_mn.sample(n=min(50000, len(df_mn)), random_state=42)
ax.scatter(sample['lon'], sample['lat'], c='blue', s=0.5, alpha=0.1)

for boundary in mn_boundaries:
    lons = boundary[:, 0]
    lats = boundary[:, 1]
    ax.plot(lons, lats, 'g-', linewidth=1.0, alpha=0.8)

ax.plot(peak_lon, peak_lat, 'r*', markersize=20, label=f'Peak: ({peak_lon:.2f}, {peak_lat:.2f})')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('2D View: Raw GPS Points + County Boundaries')
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('output/results/debug_alignment.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: output/results/debug_alignment.png")
