import shapefile
import numpy as np

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

all_lons = np.concatenate([b[:, 0] for b in mn_boundaries])
all_lats = np.concatenate([b[:, 1] for b in mn_boundaries])
import sys
print(f'Shapefile lon: {all_lons.min():.4f} to {all_lons.max():.4f}', flush=True)
print(f'Shapefile lat: {all_lats.min():.4f} to {all_lats.max():.4f}', flush=True)
print(f'Number of boundaries: {len(mn_boundaries)}', flush=True)
sys.stdout.flush()
