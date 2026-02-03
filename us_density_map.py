#!/usr/bin/env python3
"""Create a visually stunning 3D density map of cat GPS data across the entire US."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
import shapefile

plt.switch_backend('Agg')

print("Loading data...")
df = pd.read_csv('output/raw.tsv.gz', sep='\t', compression='gzip', low_memory=False)

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
hist_smooth = gaussian_filter(hist, sigma=1.0)

x_centers = (xedges[:-1] + xedges[1:]) / 2
y_centers = (yedges[:-1] + yedges[1:]) / 2
X, Y = np.meshgrid(x_centers, y_centers)

# Use power scaling to make peaks more prominent
Z_raw = np.log1p(hist_smooth.T)
Z = np.power(Z_raw, 2.0)  # Stronger exaggeration of peaks

# Create a masked array to make near-zero density areas transparent
threshold = Z.max() * 0.02  # Areas below 2% of max become transparent
Z_masked = np.ma.masked_where(Z < threshold, Z)

print(f"X (lon) range: {X.min():.4f} to {X.max():.4f}")
print(f"Y (lat) range: {Y.min():.4f} to {Y.max():.4f}")
print(f"Z range: {Z.min():.4f} to {Z.max():.4f}")

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

# Create the 3D plot
fig = plt.figure(figsize=(20, 14))
ax = fig.add_subplot(111, projection='3d')

# Plot the density surface with a nice colormap (masked areas will be transparent)
surf = ax.plot_surface(X, Y, Z_masked, cmap='inferno', alpha=0.95,
                        linewidth=0, antialiased=True,
                        rstride=1, cstride=1)

# Draw state/county boundaries at a small negative Z to be visible as floor
boundary_z = -0.3
for boundary in all_boundaries:
    lons = boundary[:, 0]
    lats = boundary[:, 1]
    # Only draw if within our bounds
    if lons.max() >= US_LON_MIN and lons.min() <= US_LON_MAX and \
       lats.max() >= US_LAT_MIN and lats.min() <= US_LAT_MAX:
        ax.plot(lons, lats, zs=boundary_z, zdir='z', color='#333333', linewidth=0.3, alpha=0.7)

# Set viewing angle - similar to Minnesota minimal view
ax.view_init(elev=75, azim=-90)

# Remove all axes, panes, and gridlines for clean look
ax.set_axis_off()
ax.set_title('Cat GPS Activity Density Across the Continental United States', fontsize=16, fontweight='bold', pad=20)

# Adjust the aspect ratio to account for latitude distortion
# At ~37° latitude (center of US), 1° lon ≈ 0.8° lat in distance
ax.set_box_aspect([1.5, 1, 0.7])  # Wider than tall, taller Z for dramatic peaks

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=15, pad=0.02)
cbar.set_label('Log Density', fontsize=11)

plt.tight_layout()
plt.savefig('output/results/us_density_3d.pdf', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ Saved: output/results/us_density_3d.pdf")

# Create additional views
views = [
    (70, -95, 'us_density_3d_angled'),
    (60, -100, 'us_density_3d_perspective'),
]

for elev, azim, name in views:
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z_masked, cmap='inferno', alpha=0.95,
                            linewidth=0, antialiased=True,
                            rstride=1, cstride=1)

    for boundary in all_boundaries:
        lons = boundary[:, 0]
        lats = boundary[:, 1]
        if lons.max() >= US_LON_MIN and lons.min() <= US_LON_MAX and \
           lats.max() >= US_LAT_MIN and lats.min() <= US_LAT_MAX:
            ax.plot(lons, lats, zs=boundary_z, zdir='z', color='#333333', linewidth=0.3, alpha=0.7)

    ax.view_init(elev=elev, azim=azim)

    # Remove all axes, panes, and gridlines for clean look
    ax.set_axis_off()

    ax.set_title('Cat GPS Activity Density Across the Continental United States', fontsize=16, fontweight='bold', pad=20)
    ax.set_box_aspect([1.5, 1, 0.7])

    cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=15, pad=0.02)
    cbar.set_label('Log Density', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'output/results/{name}.pdf', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Saved: output/results/{name}.pdf")

print("\nDone! Created 3 US density maps.")

# Create rotating GIF animation
print("\n🎬 Creating rotating animation...")
from matplotlib.animation import FuncAnimation, PillowWriter

# Create figure for animation
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the density surface with stride for faster rendering
surf = ax.plot_surface(X, Y, Z_masked, cmap='inferno', alpha=0.95,
                        linewidth=0, antialiased=True,
                        rstride=2, cstride=2)  # Stride of 2 for faster GIF

# Draw boundaries (fewer for animation speed)
for boundary in all_boundaries[::3]:  # Every 3rd boundary for speed
    lons = boundary[:, 0]
    lats = boundary[:, 1]
    if lons.max() >= US_LON_MIN and lons.min() <= US_LON_MAX and \
       lats.max() >= US_LAT_MIN and lats.min() <= US_LAT_MAX:
        ax.plot(lons, lats, zs=boundary_z, zdir='z', color='#333333', linewidth=0.3, alpha=0.7)

ax.set_axis_off()
ax.set_title('Cat GPS Activity Density Across the Continental United States', fontsize=16, fontweight='bold', pad=20)
ax.set_box_aspect([1.5, 1, 0.7])

# Add colorbar
cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=15, pad=0.02)
cbar.set_label('Log Density', fontsize=11)

plt.tight_layout()

# Animation parameters
n_frames = 180  # More frames for slower, smoother rotation (15 sec loop at 12fps)
azim_start = -90

def update(frame):
    # Full 360 rotation over all frames
    azim = azim_start + (frame / n_frames) * 360

    # Vary elevation smoothly: start high, dip low in middle, return high
    # Use sine wave to smoothly vary between 75 and 45 degrees
    elev = 60 + 15 * np.cos(2 * np.pi * frame / n_frames)

    ax.view_init(elev=elev, azim=azim)
    return []

print(f"Rendering {n_frames} frames...")
anim = FuncAnimation(fig, update, frames=n_frames, interval=83, blit=False)

# Save as high-quality GIF
writer = PillowWriter(fps=12)
anim.save('output/results/us_density_3d_rotating.gif', writer=writer, dpi=100)
plt.close()
print("✓ Saved: output/results/us_density_3d_rotating.gif")

print("\n🎉 All done! Created 3 PDFs and 1 rotating GIF.")

