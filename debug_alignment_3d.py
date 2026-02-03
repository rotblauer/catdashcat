#!/usr/bin/env python3
"""3D version of debug alignment with minimal rotation to show density surface."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
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

# Create 3D plot with minimal rotation
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the density surface
surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.9,
                        linewidth=0, antialiased=True,
                        rstride=1, cstride=1)

# Overlay boundaries slightly below Z=0 so they're visible under the surface
for boundary in mn_boundaries:
    lons = boundary[:, 0]
    lats = boundary[:, 1]
    ax.plot(lons, lats, zs=-0.5, zdir='z', color='black', linewidth=2.0, alpha=1.0)

# Mark peak density
peak_idx = np.unravel_index(Z.argmax(), Z.shape)
peak_lon = X[peak_idx]
peak_lat = Y[peak_idx]
peak_z = Z[peak_idx]
ax.scatter([peak_lon], [peak_lat], [peak_z], c='red', s=100, marker='*',
           label=f'Peak: ({peak_lon:.2f}, {peak_lat:.2f})')

# Set viewing angle - minimal rotation from top-down
# azim: rotation around vertical axis (-90 is looking from south)
# elev: elevation angle (90 is directly above, lower values tilt the view)
ax.view_init(elev=75, azim=-90)  # High elevation, slight tilt

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Log Density')
ax.set_title('3D Density Surface - Minnesota Cat GPS Data\n(Minimal rotation from top-down view)')

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Log Density')

plt.tight_layout()
plt.savefig('output/results/debug_alignment_3d_minimal.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: output/results/debug_alignment_3d_minimal.png")

# Create a few more views with slightly more rotation for comparison
views = [
    (70, -80, 'slight_tilt'),
    (60, -75, 'moderate_tilt'),
    (45, -60, 'angled_view'),
]

for elev, azim, name in views:
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.9,
                            linewidth=0, antialiased=True,
                            rstride=1, cstride=1)

    for boundary in mn_boundaries:
        lons = boundary[:, 0]
        lats = boundary[:, 1]
        ax.plot(lons, lats, zs=-0.5, zdir='z', color='black', linewidth=2.0, alpha=1.0)

    ax.scatter([peak_lon], [peak_lat], [peak_z], c='red', s=100, marker='*')
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Log Density')
    ax.set_title(f'3D Density Surface - {name.replace("_", " ").title()}\n(elev={elev}°, azim={azim}°)')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Log Density')

    plt.tight_layout()
    plt.savefig(f'output/results/debug_alignment_3d_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: output/results/debug_alignment_3d_{name}.png")

print("\nDone! Created 4 views with increasing rotation.")
