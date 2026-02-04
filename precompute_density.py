#!/usr/bin/env python3
"""Precompute density histograms at multiple resolutions for the WebGL viewer.

This script reads the raw GPS data once and outputs a compact JSON file containing
pre-binned histograms at multiple resolutions. The WebGL viewer can then apply
dynamic sigma/power transformations client-side.

Example usage:
    python precompute_density.py -i output/raw.tsv.gz -o output/density_data.json
    python precompute_density.py --resolutions 200 400 800 --chunk-size 500000
"""

import argparse
import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path

# Continental US bounds
US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -125.0, -66.5

DEFAULT_RESOLUTIONS = [100, 250, 500, 1000]
DEFAULT_CHUNK_SIZE = 500_000


def build_histograms_multi_resolution(input_file: str, resolutions: list, chunk_size: int) -> tuple:
    """Build 2D histograms at multiple resolutions in a single pass through the file.

    This is much more efficient than reading the file multiple times.
    """
    # Prepare edges and histograms for all resolutions
    histograms = {}
    edges = {}
    for res in resolutions:
        edges[res] = {
            'lon': np.linspace(US_LON_MIN, US_LON_MAX, res + 1, dtype=np.float32),
            'lat': np.linspace(US_LAT_MIN, US_LAT_MAX, res + 1, dtype=np.float32)
        }
        histograms[res] = np.zeros((res, res), dtype=np.float32)

    total_points = 0
    chunks_processed = 0

    print(f"Reading data in chunks of {chunk_size:,}...")
    print(f"Computing resolutions: {resolutions}")

    for chunk in pd.read_csv(
        input_file,
        sep='\t',
        compression='gzip',
        usecols=['lat', 'lon'],
        dtype={'lat': 'str', 'lon': 'str'},
        chunksize=chunk_size,
        on_bad_lines='skip'
    ):
        chunk['lat'] = pd.to_numeric(chunk['lat'], errors='coerce')
        chunk['lon'] = pd.to_numeric(chunk['lon'], errors='coerce')

        mask = (
            chunk['lat'].between(US_LAT_MIN, US_LAT_MAX) &
            chunk['lon'].between(US_LON_MIN, US_LON_MAX)
        )
        filtered = chunk.loc[mask].dropna()

        if filtered.empty:
            chunks_processed += 1
            continue

        lons = filtered['lon'].values
        lats = filtered['lat'].values

        # Accumulate into all histograms
        for res in resolutions:
            h, _, _ = np.histogram2d(
                lons, lats,
                bins=[edges[res]['lon'], edges[res]['lat']]
            )
            histograms[res] += h.astype(np.float32)

        total_points += len(filtered)
        chunks_processed += 1

        if chunks_processed % 10 == 0:
            print(f"    Chunk {chunks_processed}: {total_points:,} points...")

    print(f"✓ Finished: {total_points:,} points from {chunks_processed} chunks")
    return histograms, total_points


def extract_state_boundaries(shapefile_path: str) -> list:
    """Extract simplified state boundaries for overlay."""
    import shapefile
    sf = shapefile.Reader(shapefile_path)

    # Group by state FIPS and simplify
    state_polys = {}
    skip_states = {'02', '15', '72', '78', '60', '66', '69'}  # Non-continental

    for shaperec in sf.iterShapeRecords():
        record = shaperec.record
        state_fip = str(record[0]) if record else ''

        if state_fip in skip_states:
            continue

        shape = shaperec.shape
        if shape.shapeType in [5, 15]:
            parts = list(shape.parts) + [len(shape.points)]
            for i in range(len(parts) - 1):
                pts = shape.points[parts[i]:parts[i+1]]
                if len(pts) < 4:
                    continue

                coords = np.array(pts, dtype=np.float32)

                # Filter to US bounds
                if (coords[:, 0].max() < US_LON_MIN or coords[:, 0].min() > US_LON_MAX or
                    coords[:, 1].max() < US_LAT_MIN or coords[:, 1].min() > US_LAT_MAX):
                    continue

                # Simplify by keeping every Nth point (reduce data size)
                step = max(1, len(coords) // 50)
                simplified = coords[::step].tolist()

                if state_fip not in state_polys:
                    state_polys[state_fip] = []
                state_polys[state_fip].append(simplified)

    # Flatten to list of polylines
    boundaries = []
    for state, polys in state_polys.items():
        for poly in polys:
            if len(poly) >= 3:
                boundaries.append(poly)

    return boundaries


def compress_histogram(hist: np.ndarray) -> dict:
    """Compress histogram to sparse format for JSON serialization."""
    # Find non-zero cells
    nonzero = hist > 0
    indices = np.argwhere(nonzero)
    values = hist[nonzero]

    # Store as sparse representation
    return {
        'shape': list(hist.shape),
        'indices': indices.tolist(),
        'values': values.tolist(),
        'max': float(values.max()) if len(values) > 0 else 0,
        'sum': float(values.sum())
    }


def main():
    parser = argparse.ArgumentParser(description='Precompute density histograms for WebGL viewer')
    parser.add_argument('-i', '--input', default='output/raw.tsv.gz',
                        help='Input TSV file')
    parser.add_argument('-o', '--output', default='output/density_data.json.gz',
                        help='Output JSON file (gzipped)')
    parser.add_argument('--resolutions', type=int, nargs='+', default=DEFAULT_RESOLUTIONS,
                        help=f'Resolutions to compute (default: {DEFAULT_RESOLUTIONS})')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Rows per chunk (default: {DEFAULT_CHUNK_SIZE:,})')
    parser.add_argument('--shapefile', default='output/cb_2020_us_county_500k.shp',
                        help='Shapefile for state boundaries')
    args = parser.parse_args()

    output_data = {
        'bounds': {
            'lat_min': US_LAT_MIN,
            'lat_max': US_LAT_MAX,
            'lon_min': US_LON_MIN,
            'lon_max': US_LON_MAX
        },
        'histograms': {},
        'boundaries': [],
        'total_points': 0
    }

    # Compute all histograms in a single pass through the file
    print(f"\n📊 Computing histograms at {len(args.resolutions)} resolutions in single pass...")
    histograms, total_points = build_histograms_multi_resolution(
        args.input, args.resolutions, args.chunk_size
    )
    output_data['total_points'] = total_points

    for res in sorted(args.resolutions):
        hist = histograms[res]
        output_data['histograms'][str(res)] = compress_histogram(hist)
        print(f"   ✓ {res}x{res}: {(hist > 0).sum():,} non-zero cells, max={hist.max():.0f}")

    # Extract boundaries
    print(f"\n🗺️  Extracting state boundaries...")
    try:
        boundaries = extract_state_boundaries(args.shapefile)
        output_data['boundaries'] = boundaries
        print(f"   ✓ {len(boundaries)} boundary polygons")
    except Exception as e:
        print(f"   ⚠ Could not load boundaries: {e}")

    # Write compressed JSON
    print(f"\n💾 Writing {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(output_data, separators=(',', ':'))

    if args.output.endswith('.gz'):
        with gzip.open(args.output, 'wt', encoding='utf-8') as f:
            f.write(json_str)
    else:
        with open(args.output, 'w') as f:
            f.write(json_str)

    # Report size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Output size: {size_mb:.2f} MB")

    print(f"\n🎉 Done! Precomputed {len(args.resolutions)} resolution levels.")
    print(f"   Total points: {output_data['total_points']:,}")


if __name__ == '__main__':
    main()
