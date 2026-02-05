#!/usr/bin/env python3
"""Generate a self-contained static HTML globe viewer with embedded density data.

This creates a single HTML file with an interactive 3D globe that can be opened
directly in a browser without needing a web server.

Example usage:
    python generate_globe_viewer.py -i output/raw.tsv.gz -o output/viewer/globe_viewer.html
    python generate_globe_viewer.py --resolutions 100 250 500 --sigma 1.2 --power 2.5
    .venv/bin/python generate_globe_viewer.py --resolutions 180 360 720
    .venv/bin/python generate_globe_viewer.py --resolutions 2880 --sigma 0.01  --n-peaks 10 --local-resolution 600 --workers 10

"""

import argparse
import base64
import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# World bounds
WORLD_LAT_MIN, WORLD_LAT_MAX = -90.0, 90.0
WORLD_LON_MIN, WORLD_LON_MAX = -180.0, 180.0

DEFAULT_RESOLUTIONS = [180, 360, 720, 1440, 2880]  # Up to 0.0625° resolution for fine detail
DEFAULT_CHUNK_SIZE = 500_000


def build_histograms_multi_resolution(input_file: str, resolutions: list, chunk_size: int) -> tuple:
    """Build 2D histograms at multiple resolutions in a single pass (global extent)."""
    histograms = {}
    edges = {}
    for res in resolutions:
        # For globe: longitude wraps, latitude doesn't
        # res determines the number of bins (e.g., 360 = 1° bins)
        edges[res] = {
            'lon': np.linspace(WORLD_LON_MIN, WORLD_LON_MAX, res + 1, dtype=np.float32),
            'lat': np.linspace(WORLD_LAT_MIN, WORLD_LAT_MAX, res // 2 + 1, dtype=np.float32)
        }
        histograms[res] = np.zeros((res, res // 2), dtype=np.float32)

    total_points = 0
    chunks_processed = 0

    print(f"Reading data in chunks of {chunk_size:,}...")

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

        # Filter valid coordinates
        mask = (
            chunk['lat'].between(WORLD_LAT_MIN, WORLD_LAT_MAX) &
            chunk['lon'].between(WORLD_LON_MIN, WORLD_LON_MAX)
        )
        filtered = chunk.loc[mask].dropna()

        if filtered.empty:
            chunks_processed += 1
            continue

        lons = filtered['lon'].values
        lats = filtered['lat'].values

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


def extract_us_boundaries(shapefile_path: str, simplify_factor: int = 100) -> list:
    """Extract simplified US state/county boundaries for overlay."""
    try:
        import shapefile
        sf = shapefile.Reader(shapefile_path)

        boundaries = []
        skip_states = {'02', '15', '72', '78', '60', '66', '69'}  # Non-continental US

        for shaperec in sf.iterShapeRecords():
            record = shaperec.record
            state_fip = str(record[0]) if record else ''

            if state_fip in skip_states:
                continue

            shape = shaperec.shape
            if shape.shapeType in [5, 15]:  # Polygon types
                parts = list(shape.parts) + [len(shape.points)]
                for i in range(len(parts) - 1):
                    pts = shape.points[parts[i]:parts[i+1]]
                    if len(pts) < 4:
                        continue

                    coords = np.array(pts, dtype=np.float32)

                    # Simplify by keeping every Nth point
                    step = max(1, len(coords) // simplify_factor)
                    simplified = coords[::step].tolist()

                    if len(simplified) >= 3:
                        boundaries.append(simplified)

        return boundaries
    except Exception as e:
        print(f"   ⚠ Could not load US boundaries: {e}")
        return []


def get_world_boundaries() -> list:
    """Get simplified world country boundaries.

    These are embedded simplified outlines of major landmasses for globe visualization.
    """
    # Try to load from Natural Earth shapefile if available
    try:
        import shapefile
        import os

        # Check common locations for Natural Earth data
        ne_paths = [
            'output/ne_110m_admin_0_countries.shp',
            'ne_110m_admin_0_countries.shp',
            '/usr/share/naturalearth/ne_110m_admin_0_countries.shp'
        ]

        for ne_path in ne_paths:
            if os.path.exists(ne_path):
                print(f"   Loading world boundaries from {ne_path}")
                sf = shapefile.Reader(ne_path)
                boundaries = []

                for shaperec in sf.iterShapeRecords():
                    shape = shaperec.shape
                    if shape.shapeType in [5, 15]:
                        parts = list(shape.parts) + [len(shape.points)]
                        for i in range(len(parts) - 1):
                            pts = shape.points[parts[i]:parts[i+1]]
                            if len(pts) < 4:
                                continue
                            coords = np.array(pts, dtype=np.float32)
                            # Simplify more aggressively for world view
                            step = max(1, len(coords) // 30)
                            simplified = coords[::step].tolist()
                            if len(simplified) >= 3:
                                boundaries.append(simplified)

                return boundaries
    except Exception as e:
        print(f"   Natural Earth not available: {e}")

    # Return empty if no world boundaries available
    # User can download Natural Earth data for full boundaries
    print("   ℹ️  For world boundaries, download Natural Earth:")
    print("      https://www.naturalearthdata.com/downloads/110m-cultural-vectors/")
    return []


def compress_histogram(hist: np.ndarray) -> dict:
    """Compress histogram to sparse format."""
    nonzero = hist > 0
    indices = np.argwhere(nonzero)
    values = hist[nonzero]

    return {
        'shape': list(hist.shape),
        'indices': indices.tolist(),
        'values': values.tolist(),
        'max': float(values.max()) if len(values) > 0 else 0,
        'sum': float(values.sum())
    }


def find_top_peaks(histograms: dict, n_peaks: int = 10) -> list:
    """Find the top N peaks from the highest resolution histogram."""
    # Use the highest resolution histogram
    max_res = max(histograms.keys())
    hist = histograms[max_res]
    width, height = hist.shape

    # Get top N cells by count
    flat_indices = np.argsort(hist.flatten())[::-1][:n_peaks * 3]  # Get extra to filter duplicates

    peaks = []
    min_distance_deg = 1.0  # Minimum 1 degree separation between peaks

    for flat_idx in flat_indices:
        if len(peaks) >= n_peaks:
            break

        x = flat_idx // height
        y = flat_idx % height
        count = hist[x, y]

        if count <= 0:
            continue

        # Convert to lat/lon
        lon = (x + 0.5) / width * 360 - 180
        lat = (y + 0.5) / height * 180 - 90

        # Check distance from existing peaks
        too_close = False
        for existing in peaks:
            dlat = abs(lat - existing['lat'])
            dlon = abs(lon - existing['lon'])
            if dlat < min_distance_deg and dlon < min_distance_deg:
                too_close = True
                break

        if not too_close:
            peaks.append({
                'lat': float(lat),
                'lon': float(lon),
                'count': int(count),
                'rank': len(peaks) + 1
            })

    return peaks


def build_local_histogram(input_file: str, center_lat: float, center_lon: float,
                          radius_km: float = 50, resolution: int = 200,
                          chunk_size: int = 500_000, peak_index: int = 0) -> dict:
    """Build a high-resolution histogram for a local region around a point.

    Args:
        center_lat, center_lon: Center of the region
        radius_km: Radius in kilometers (default 50km = 100km x 100km region)
        resolution: Number of bins per side
        peak_index: Index of the peak (for logging)
    """
    # Convert km to approximate degrees (at given latitude)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat))

    lat_radius = radius_km / km_per_deg_lat
    lon_radius = radius_km / km_per_deg_lon

    lat_min = center_lat - lat_radius
    lat_max = center_lat + lat_radius
    lon_min = center_lon - lon_radius
    lon_max = center_lon + lon_radius

    lon_edges = np.linspace(lon_min, lon_max, resolution + 1, dtype=np.float32)
    lat_edges = np.linspace(lat_min, lat_max, resolution + 1, dtype=np.float32)
    hist = np.zeros((resolution, resolution), dtype=np.float32)

    total_points = 0

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
            chunk['lat'].between(lat_min, lat_max) &
            chunk['lon'].between(lon_min, lon_max)
        )
        filtered = chunk.loc[mask].dropna()

        if filtered.empty:
            continue

        h, _, _ = np.histogram2d(
            filtered['lon'].values,
            filtered['lat'].values,
            bins=[lon_edges, lat_edges]
        )
        hist += h.astype(np.float32)
        total_points += len(filtered)

    return {
        'peak_index': peak_index,
        'histogram': compress_histogram(hist),
        'bounds': {
            'lat_min': float(lat_min),
            'lat_max': float(lat_max),
            'lon_min': float(lon_min),
            'lon_max': float(lon_max)
        },
        'center': {'lat': center_lat, 'lon': center_lon},
        'radius_km': radius_km,
        'total_points': total_points
    }


def build_local_histograms_parallel(input_file: str, peaks: list, radius_km: float = 50,
                                     resolution: int = 200, chunk_size: int = 500_000,
                                     max_workers: int = 4) -> dict:
    """Build local histograms for all peaks in parallel using threads.

    Args:
        input_file: Path to input TSV file
        peaks: List of peak dictionaries with 'lat' and 'lon'
        radius_km: Radius in km for each local region
        resolution: Number of bins per side
        chunk_size: Rows per chunk when reading
        max_workers: Number of parallel threads

    Returns:
        Dictionary mapping peak index (as string) to local histogram data
    """
    local_views = {}
    print_lock = threading.Lock()

    def process_peak(args):
        idx, peak = args
        result = build_local_histogram(
            input_file,
            peak['lat'],
            peak['lon'],
            radius_km=radius_km,
            resolution=resolution,
            chunk_size=chunk_size,
            peak_index=idx
        )
        with print_lock:
            print(f"   ✓ Peak {idx+1}: ({peak['lat']:.2f}, {peak['lon']:.2f}) - {result['total_points']:,} points")
        return idx, result

    # Use ThreadPoolExecutor for parallel I/O
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_peak, (i, peak)) for i, peak in enumerate(peaks)]

        for future in as_completed(futures):
            idx, result = future.result()
            local_views[str(idx)] = result

    return local_views


def generate_html(density_data: dict, default_sigma: float = 1.0, default_power: float = 2.0) -> str:
    """Generate the complete static HTML globe viewer with embedded data."""

    # Compress and encode the density data
    json_str = json.dumps(density_data, separators=(',', ':'))
    compressed = gzip.compress(json_str.encode('utf-8'))
    encoded_data = base64.b64encode(compressed).decode('ascii')

    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cat GPS Density Globe - Interactive 3D Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: radial-gradient(ellipse at center, #1a1a2e 0%, #0d0d1a 100%);
            color: #fff;
            overflow: hidden;
        }
        
        #container { width: 100vw; height: 100vh; position: relative; }
        #canvas-container { width: 100%; height: 100%; }
        
        #controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(20, 20, 40, 0.9);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            min-width: 260px;
            z-index: 100;
        }
        
        #controls h1 { font-size: 18px; margin-bottom: 5px; color: #feca57; }
        #controls .subtitle { font-size: 12px; color: #888; margin-bottom: 20px; }
        
        .control-group { margin-bottom: 16px; }
        .control-group label {
            display: block;
            font-size: 11px;
            color: #aaa;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .control-group input[type="range"] {
            width: 100%;
            height: 5px;
            border-radius: 3px;
            background: #333;
            outline: none;
            -webkit-appearance: none;
        }
        
        .control-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #e94560;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(233, 69, 96, 0.4);
        }
        
        .control-group .value { font-size: 13px; color: #fff; font-weight: 600; margin-top: 3px; }
        
        .control-group select {
            width: 100%;
            padding: 7px 10px;
            border-radius: 6px;
            background: #2a2a3a;
            border: 1px solid #444;
            color: #fff;
            font-size: 13px;
            cursor: pointer;
        }
        
        .control-group select:focus { outline: none; border-color: #e94560; }
        
        #stats {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(20, 20, 40, 0.85);
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 11px;
            color: #888;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        #stats span { color: #feca57; font-weight: 600; }
        
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            z-index: 200;
        }
        
        #loading .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255, 255, 255, 0.1);
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        #loading p { color: #aaa; font-size: 14px; }
        .hidden { display: none !important; }
        
        #colorbar {
            position: absolute;
            right: 25px;
            top: 50%;
            transform: translateY(-50%);
            width: 16px;
            height: 180px;
            border-radius: 4px;
            background: linear-gradient(to top,
                #283c64 0%, #3c64a0 15%, #64a0dc 30%,
                #c85078 45%, #ff5a5a 60%, #ffa03c 75%, #ffdc50 90%, #ffffc8 100%
            );
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        #colorbar-labels {
            position: absolute;
            right: 50px;
            top: 50%;
            transform: translateY(-50%);
            height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            font-size: 10px;
            color: #888;
        }
        
        .hint { font-size: 9px; color: #666; margin-top: 2px; font-style: italic; }
        
        #focus-btn, #side-view-btn, #reset-view-btn {
            width: 100%;
            padding: 8px;
            margin-top: 8px;
            background: #e94560;
            border: none;
            border-radius: 6px;
            color: white;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        #focus-btn:hover, #side-view-btn:hover, #reset-view-btn:hover { background: #ff6b6b; }
        
        #side-view-btn { background: #4a90d9; }
        #side-view-btn:hover { background: #6ba8e8; }
        
        #reset-view-btn { background: #666; }
        #reset-view-btn:hover { background: #888; }
        
        /* Peak labels panel */
        #peaks-panel {
            position: absolute;
            top: 20px;
            right: 80px;
            background: rgba(20, 20, 40, 0.9);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-width: 200px;
            z-index: 100;
        }
        
        #peaks-panel h3 { font-size: 14px; color: #feca57; margin-bottom: 10px; }
        
        .peak-item {
            display: flex;
            align-items: center;
            padding: 6px 8px;
            margin: 4px 0;
            background: rgba(255,255,255,0.05);
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 11px;
        }
        
        .peak-item:hover { background: rgba(233, 69, 96, 0.3); }
        .peak-item.active { background: rgba(233, 69, 96, 0.5); border: 1px solid #e94560; }
        
        .peak-rank {
            width: 20px;
            height: 20px;
            background: #e94560;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 10px;
            margin-right: 8px;
        }
        
        .peak-info { flex: 1; }
        .peak-info .coords { color: #888; font-size: 9px; }
        
        /* Local view overlay */
        #local-view-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0,0,0,0.9);
            z-index: 500;
            display: none;
        }
        
        #local-view-overlay.active { display: flex; }
        
        #local-view-container {
            display: flex;
            width: 100%;
            height: 100%;
        }
        
        #local-map-container {
            flex: 1;
            position: relative;
        }
        
        #local-3d-container {
            flex: 1;
            position: relative;
        }
        
        #local-controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(20, 20, 40, 0.95);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            min-width: 220px;
            z-index: 510;
        }
        
        #local-controls h2 { font-size: 16px; color: #feca57; margin-bottom: 5px; }
        #local-controls .subtitle { font-size: 11px; color: #888; margin-bottom: 15px; }
        
        #close-local-btn {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 40px;
            height: 40px;
            background: #e94560;
            border: none;
            border-radius: 50%;
            color: white;
            font-size: 20px;
            cursor: pointer;
            z-index: 520;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        #close-local-btn:hover { background: #ff6b6b; }
        
        /* Leaflet map styling */
        #local-map {
            width: 100%;
            height: 100%;
        }
        
        .leaflet-container { background: #1a1a2e; }
        
        /* Google Earth-style Navigation Widget */
        #nav-widget {
            position: absolute;
            bottom: 100px;
            right: 25px;
            z-index: 100;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        
        .nav-compass {
            width: 90px;
            height: 90px;
            position: relative;
        }
        
        .nav-ring {
            width: 90px;
            height: 90px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            position: absolute;
            background: rgba(20, 20, 40, 0.7);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        
        .nav-n {
            position: absolute;
            top: -8px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 10px;
            font-weight: bold;
            color: #e94560;
        }
        
        .nav-btn {
            position: absolute;
            width: 28px;
            height: 28px;
            background: rgba(40, 40, 60, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            color: #fff;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s;
        }
        
        .nav-btn:hover {
            background: rgba(233, 69, 96, 0.6);
            border-color: #e94560;
        }
        
        .nav-btn:active {
            transform: scale(0.95);
            background: rgba(233, 69, 96, 0.8);
        }
        
        .nav-up { top: 4px; left: 50%; transform: translateX(-50%); }
        .nav-down { bottom: 4px; left: 50%; transform: translateX(-50%); }
        .nav-left { left: 4px; top: 50%; transform: translateY(-50%); }
        .nav-right { right: 4px; top: 50%; transform: translateY(-50%); }
        
        .nav-center {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 24px;
            height: 24px;
            background: rgba(60, 60, 80, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
        }
        
        .nav-center:hover { background: rgba(233, 69, 96, 0.6); }
        
        .nav-zoom {
            display: flex;
            flex-direction: column;
            background: rgba(20, 20, 40, 0.8);
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        
        .zoom-btn {
            width: 36px;
            height: 36px;
            background: transparent;
            border: none;
            color: #fff;
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s;
        }
        
        .zoom-btn:hover { background: rgba(233, 69, 96, 0.4); }
        .zoom-btn:active { background: rgba(233, 69, 96, 0.7); }
        
        .zoom-divider {
            width: 24px;
            height: 1px;
            background: rgba(255, 255, 255, 0.2);
            margin: 0 auto;
        }
        
        .nav-tilt {
            display: flex;
            flex-direction: column;
            background: rgba(20, 20, 40, 0.8);
            border-radius: 15px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        
        .tilt-btn {
            width: 30px;
            height: 26px;
            background: transparent;
            border: none;
            color: #fff;
            font-size: 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.15s;
        }
        
        .tilt-btn:hover { background: rgba(233, 69, 96, 0.4); }
        .tilt-btn:active { background: rgba(233, 69, 96, 0.7); }
        
        .nav-label {
            font-size: 8px;
            color: #888;
            text-align: center;
            margin-top: 2px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
    </style>
    
    <!-- Leaflet CSS for map -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
</head>
<body>
    <div id="container">
        <div id="canvas-container"></div>
        
        <div id="loading">
            <div class="spinner"></div>
            <p>Loading globe data...</p>
        </div>
        
        <div id="controls" class="hidden">
            <h1>🌍 Cat GPS Globe</h1>
            <p class="subtitle">Worldwide Distribution</p>
            
            <div class="control-group">
                <label>Resolution</label>
                <select id="resolution"></select>
            </div>
            
            <div class="control-group">
                <label>Smoothing (σ)</label>
                <input type="range" id="sigma" min="0" max="0.5" step="0.01" value="''' + str(default_sigma) + '''">
                <div class="value" id="sigma-value">''' + str(default_sigma) + '''</div>
                <div class="hint">0 = sharp peaks, 0.5 = very smooth</div>
            </div>
            
            <div class="control-group">
                <label>Peak Height</label>
                <input type="range" id="power" min="1" max="4" step="0.1" value="''' + str(default_power) + '''">
                <div class="value" id="power-value">''' + str(default_power) + '''</div>
                <div class="hint">Higher = taller peaks</div>
            </div>
            
            <div class="control-group">
                <label>Extrusion Scale</label>
                <input type="range" id="height-scale" min="0.05" max="0.8" step="0.01" value="0.25">
                <div class="value" id="height-scale-value">0.25</div>
                <div class="hint">Height of peaks</div>
            </div>
            
            <div class="control-group">
                <label>Threshold</label>
                <input type="range" id="threshold" min="0" max="0.2" step="0.005" value="0.02">
                <div class="value" id="threshold-value">0.02</div>
                <div class="hint">Min density to show</div>
            </div>
            
            <div class="control-group">
                <label>Show Boundaries</label>
                <select id="show-boundaries">
                    <option value="true" selected>Yes</option>
                    <option value="false">No</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Auto Rotate</label>
                <select id="auto-rotate">
                    <option value="false">Off</option>
                    <option value="true" selected>On</option>
                </select>
            </div>
            
            <div class="control-group">
                <label>Base Map</label>
                <select id="base-map">
                    <option value="dark">Dark</option>
                    <option value="natural" selected>Natural Earth</option>
                    <option value="topo">Topography</option>
                </select>
            </div>
            
            <button id="focus-btn">🇺🇸 Focus on USA</button>
            <button id="side-view-btn">↔️ Side View</button>
            <button id="reset-view-btn">🔄 Reset View</button>
        </div>
        
        <div id="peaks-panel" class="hidden">
            <h3>🏔️ Top Peaks</h3>
            <div id="peaks-list"></div>
        </div>
        
        <div id="stats" class="hidden">
            GPS observations: <span id="point-count">0</span> |
            Drag to rotate, scroll to zoom, right-drag to pan
        </div>
        
        <div id="colorbar" class="hidden"></div>
        <div id="colorbar-labels" class="hidden">
            <span>High</span>
            <span>Low</span>
        </div>
        
        <!-- Google Earth-style Navigation Widget -->
        <div id="nav-widget" class="hidden">
            <div class="nav-compass">
                <div class="nav-ring"></div>
                <span class="nav-n">N</span>
                <button class="nav-btn nav-up" id="nav-up" title="Rotate Up">▲</button>
                <button class="nav-btn nav-down" id="nav-down" title="Rotate Down">▼</button>
                <button class="nav-btn nav-left" id="nav-left" title="Rotate Left">◀</button>
                <button class="nav-btn nav-right" id="nav-right" title="Rotate Right">▶</button>
                <button class="nav-center" id="nav-reset" title="Reset View">⟲</button>
            </div>
            <div class="nav-zoom">
                <button class="zoom-btn" id="zoom-in" title="Zoom In">+</button>
                <div class="zoom-divider"></div>
                <button class="zoom-btn" id="zoom-out" title="Zoom Out">−</button>
            </div>
            <div class="nav-tilt">
                <button class="tilt-btn" id="tilt-up" title="Tilt Up">↑</button>
                <button class="tilt-btn" id="tilt-down" title="Tilt Down">↓</button>
            </div>
            <div class="nav-label">Tilt</div>
        </div>
    </div>
    
    <!-- Local View Overlay -->
    <div id="local-view-overlay">
        <button id="close-local-btn">✕</button>
        <div id="local-view-container">
            <div id="local-map-container">
                <div id="local-map"></div>
            </div>
            <div id="local-3d-container">
                <div id="local-canvas"></div>
            </div>
        </div>
        <div id="local-controls">
            <h2 id="local-title">Peak #1</h2>
            <p class="subtitle" id="local-subtitle">100km × 100km region</p>
            
            <div class="control-group">
                <label>Smoothing (σ)</label>
                <input type="range" id="local-sigma" min="0" max="0.5" step="0.01" value="0.05">
                <div class="value" id="local-sigma-value">0.05</div>
            </div>
            
            <div class="control-group">
                <label>Peak Height</label>
                <input type="range" id="local-power" min="1" max="4" step="0.1" value="2.0">
                <div class="value" id="local-power-value">2.0</div>
            </div>
            
            <div class="control-group">
                <label>Height Scale</label>
                <input type="range" id="local-height" min="0.1" max="2" step="0.05" value="0.5">
                <div class="value" id="local-height-value">0.50</div>
            </div>
            
            <div class="control-group">
                <label>Threshold</label>
                <input type="range" id="local-threshold" min="0" max="0.1" step="0.002" value="0.01">
                <div class="value" id="local-threshold-value">0.010</div>
            </div>
            
            <div class="control-group">
                <label>Show Map Overlay</label>
                <select id="local-map-overlay">
                    <option value="false" selected>Off</option>
                    <option value="true">On (OpenStreetMap)</option>
                </select>
            </div>
        </div>
    </div>

    <!-- Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.134.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
    <!-- Leaflet JS for map -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        // Embedded density data (base64 gzipped JSON)
        const EMBEDDED_DATA = "''' + encoded_data + '''";
        
        let scene, camera, renderer, controls;
        let globeMesh, densityMesh, boundaryLines, atmosphereMesh;
        let densityData = null;
        
        const GLOBE_RADIUS = 5;
        
        const settings = {
            resolution: 360,
            sigma: ''' + str(default_sigma) + ''',
            power: ''' + str(default_power) + ''',
            heightScale: 0.25,  // Taller peaks by default
            threshold: 0.02,    // Minimum density to show
            showBoundaries: true,
            autoRotate: true,   // Auto-rotate on by default
            baseMap: 'natural'  // 'dark', 'natural', 'topo'
        };
        
        // Color gradient for density - vibrant peaks on dark background
        const colorStops = [
            { pos: 0.00, color: [40, 60, 100] },    // Dark blue base
            { pos: 0.15, color: [60, 100, 160] },   // Medium blue
            { pos: 0.30, color: [100, 160, 220] },  // Light blue
            { pos: 0.45, color: [200, 80, 120] },   // Rose
            { pos: 0.60, color: [255, 90, 90] },    // Bright red
            { pos: 0.75, color: [255, 160, 60] },   // Orange
            { pos: 0.90, color: [255, 220, 80] },   // Yellow
            { pos: 1.00, color: [255, 255, 200] }   // Bright cream
        ];
        
        function interpolateColor(t) {
            t = Math.max(0, Math.min(1, t));
            for (let i = 0; i < colorStops.length - 1; i++) {
                if (t >= colorStops[i].pos && t <= colorStops[i + 1].pos) {
                    const localT = (t - colorStops[i].pos) / (colorStops[i + 1].pos - colorStops[i].pos);
                    const c1 = colorStops[i].color;
                    const c2 = colorStops[i + 1].color;
                    return [
                        Math.round(c1[0] + (c2[0] - c1[0]) * localT),
                        Math.round(c1[1] + (c2[1] - c1[1]) * localT),
                        Math.round(c1[2] + (c2[2] - c1[2]) * localT)
                    ];
                }
            }
            return colorStops[colorStops.length - 1].color;
        }
        
        // Convert lat/lon to 3D position on sphere
        // Standard geographic: lat=0 at equator, lon=0 at prime meridian
        function latLonToVector3(lat, lon, radius) {
            const latRad = lat * Math.PI / 180;
            const lonRad = -lon * Math.PI / 180; // Negate for correct east/west
            return new THREE.Vector3(
                radius * Math.cos(latRad) * Math.cos(lonRad),
                radius * Math.sin(latRad),
                radius * Math.cos(latRad) * Math.sin(lonRad)
            );
        }
        
        function gaussianBlur(data, width, height, sigma) {
            // Sigma slider is 0-0.5. Scale to reasonable pixel values:
            // 0 = no blur, 0.5 = moderate blur (~2% of width)
            // This gives 0-58 pixels on a 2880-wide histogram
            const pixelSigma = sigma * width * 0.04;
            
            if (pixelSigma < 0.5) return data.slice();  // Skip if less than half a pixel
            
            const kernelSize = Math.min(101, Math.ceil(pixelSigma * 3) * 2 + 1);  // Cap kernel size
            const kernel = [];
            const half = Math.floor(kernelSize / 2);
            let sum = 0;
            
            for (let i = 0; i < kernelSize; i++) {
                const x = i - half;
                const g = Math.exp(-(x * x) / (2 * pixelSigma * pixelSigma));
                kernel.push(g);
                sum += g;
            }
            for (let i = 0; i < kernelSize; i++) kernel[i] /= sum;
            
            // Horizontal pass (wrap around for longitude)
            const temp = new Float32Array(width * height);
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    let val = 0;
                    for (let k = 0; k < kernelSize; k++) {
                        let sx = x + k - half;
                        // Wrap longitude
                        if (sx < 0) sx += width;
                        if (sx >= width) sx -= width;
                        val += data[y * width + sx] * kernel[k];
                    }
                    temp[y * width + x] = val;
                }
            }
            
            // Vertical pass (clamp at poles)
            const result = new Float32Array(width * height);
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    let val = 0;
                    for (let k = 0; k < kernelSize; k++) {
                        const sy = Math.min(height - 1, Math.max(0, y + k - half));
                        val += temp[sy * width + x] * kernel[k];
                    }
                    result[y * width + x] = val;
                }
            }
            
            return result;
        }
        
        function decompressHistogram(histData) {
            const [width, height] = histData.shape;
            const data = new Float32Array(width * height);
            
            for (let i = 0; i < histData.indices.length; i++) {
                const [x, y] = histData.indices[i];
                data[y * width + x] = histData.values[i];
            }
            
            return { data, width, height };
        }
        
        function processHistogram(histData, sigma, power) {
            const { data, width, height } = decompressHistogram(histData);
            const blurred = gaussianBlur(data, width, height, sigma);
            const result = new Float32Array(width * height);
            let maxVal = 0;
            
            for (let i = 0; i < blurred.length; i++) {
                if (blurred[i] > 0) {
                    result[i] = Math.pow(Math.log1p(blurred[i]), power);
                    maxVal = Math.max(maxVal, result[i]);
                }
            }
            
            if (maxVal > 0) {
                for (let i = 0; i < result.length; i++) {
                    result[i] /= maxVal;
                }
            }
            
            return { data: result, width, height };
        }
        
        // Base map texture URLs (free tile services)
        const BASE_MAP_TEXTURES = {
            dark: null, // Solid color
            natural: 'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
            topo: 'https://unpkg.com/three-globe/example/img/earth-topology.png'
        };
        
        function createGlobe(baseMapType = 'dark') {
            const geometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
            
            if (baseMapType === 'dark' || !BASE_MAP_TEXTURES[baseMapType]) {
                // Dark ocean (default)
                const material = new THREE.MeshPhongMaterial({
                    color: 0x0a0a1a,
                    shininess: 5
                });
                return new THREE.Mesh(geometry, material);
            } else {
                // Load texture
                const textureLoader = new THREE.TextureLoader();
                const material = new THREE.MeshPhongMaterial({
                    color: 0xffffff,
                    shininess: 5,
                    transparent: true,
                    opacity: 0.7
                });
                
                textureLoader.load(
                    BASE_MAP_TEXTURES[baseMapType],
                    (texture) => {
                        material.map = texture;
                        material.needsUpdate = true;
                    },
                    undefined,
                    (err) => console.warn('Could not load base map texture:', err)
                );
                
                return new THREE.Mesh(geometry, material);
            }
        }
        
        function updateGlobeBaseMap(baseMapType) {
            if (globeMesh) {
                scene.remove(globeMesh);
                if (globeMesh.material.map) globeMesh.material.map.dispose();
                globeMesh.material.dispose();
                globeMesh.geometry.dispose();
            }
            globeMesh = createGlobe(baseMapType);
            scene.add(globeMesh);
        }
        
        function createAtmosphere() {
            const geometry = new THREE.SphereGeometry(GLOBE_RADIUS * 1.02, 64, 64);
            const material = new THREE.ShaderMaterial({
                vertexShader: `
                    varying vec3 vNormal;
                    void main() {
                        vNormal = normalize(normalMatrix * normal);
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    varying vec3 vNormal;
                    void main() {
                        float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
                        gl_FragColor = vec4(0.3, 0.6, 1.0, 1.0) * intensity * 0.4;
                    }
                `,
                blending: THREE.AdditiveBlending,
                side: THREE.BackSide,
                transparent: true
            });
            return new THREE.Mesh(geometry, material);
        }
        
        function createDensityMesh(processed) {
            const { data, width, height } = processed;
            
            const geometry = new THREE.BufferGeometry();
            const vertices = [];
            const colors = [];
            const indices = [];
            
            const heightScale = settings.heightScale * GLOBE_RADIUS;
            const threshold = settings.threshold; // Use setting for minimum value to show
            
            let vertexCount = 0;
            
            // Create individual column/spike for each cell with data
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    const z = data[idx] || 0;
                    
                    if (z < threshold) continue; // Skip empty/low cells
                    
                    // Convert grid position to lat/lon (cell center)
                    const lon = (x + 0.5) / width * 360 - 180;
                    const lat = (y + 0.5) / height * 180 - 90;
                    
                    // Cell size in degrees
                    const dLon = 360 / width;
                    const dLat = 180 / height;
                    
                    // Create a column/prism from base to peak
                    const baseRadius = GLOBE_RADIUS * 1.001;
                    const peakRadius = GLOBE_RADIUS + z * heightScale;
                    
                    // 4 corners of the cell (base)
                    const lon0 = lon - dLon/2, lon1 = lon + dLon/2;
                    const lat0 = lat - dLat/2, lat1 = lat + dLat/2;
                    
                    // Base corners (on globe surface)
                    const b0 = latLonToVector3(lat0, lon0, baseRadius);
                    const b1 = latLonToVector3(lat0, lon1, baseRadius);
                    const b2 = latLonToVector3(lat1, lon1, baseRadius);
                    const b3 = latLonToVector3(lat1, lon0, baseRadius);
                    
                    // Peak corners (extruded outward)
                    const p0 = latLonToVector3(lat0, lon0, peakRadius);
                    const p1 = latLonToVector3(lat0, lon1, peakRadius);
                    const p2 = latLonToVector3(lat1, lon1, peakRadius);
                    const p3 = latLonToVector3(lat1, lon0, peakRadius);
                    
                    // Color based on height
                    const color = interpolateColor(z);
                    const r = color[0] / 255, g = color[1] / 255, b = color[2] / 255;
                    
                    // Slightly darker base color
                    const darkFactor = 0.6;
                    const rd = r * darkFactor, gd = g * darkFactor, bd = b * darkFactor;
                    
                    const baseIdx = vertexCount;
                    
                    // Add 8 vertices: 4 base + 4 peak
                    // Base vertices (darker)
                    vertices.push(b0.x, b0.y, b0.z); colors.push(rd, gd, bd);
                    vertices.push(b1.x, b1.y, b1.z); colors.push(rd, gd, bd);
                    vertices.push(b2.x, b2.y, b2.z); colors.push(rd, gd, bd);
                    vertices.push(b3.x, b3.y, b3.z); colors.push(rd, gd, bd);
                    // Peak vertices (full color)
                    vertices.push(p0.x, p0.y, p0.z); colors.push(r, g, b);
                    vertices.push(p1.x, p1.y, p1.z); colors.push(r, g, b);
                    vertices.push(p2.x, p2.y, p2.z); colors.push(r, g, b);
                    vertices.push(p3.x, p3.y, p3.z); colors.push(r, g, b);
                    
                    // Top face (2 triangles)
                    indices.push(baseIdx+4, baseIdx+5, baseIdx+6);
                    indices.push(baseIdx+4, baseIdx+6, baseIdx+7);
                    
                    // Side faces (each side = 2 triangles)
                    // Front (lat0)
                    indices.push(baseIdx+0, baseIdx+4, baseIdx+1);
                    indices.push(baseIdx+1, baseIdx+4, baseIdx+5);
                    // Right (lon1)
                    indices.push(baseIdx+1, baseIdx+5, baseIdx+2);
                    indices.push(baseIdx+2, baseIdx+5, baseIdx+6);
                    // Back (lat1)
                    indices.push(baseIdx+2, baseIdx+6, baseIdx+3);
                    indices.push(baseIdx+3, baseIdx+6, baseIdx+7);
                    // Left (lon0)
                    indices.push(baseIdx+3, baseIdx+7, baseIdx+0);
                    indices.push(baseIdx+0, baseIdx+7, baseIdx+4);
                    
                    vertexCount += 8;
                }
            }
            
            if (vertices.length === 0) {
                // Return empty mesh if no data
                const emptyGeo = new THREE.BufferGeometry();
                return new THREE.Mesh(emptyGeo, new THREE.MeshBasicMaterial());
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            geometry.setIndex(indices);
            geometry.computeVertexNormals();
            
            const material = new THREE.MeshPhongMaterial({
                vertexColors: true,
                side: THREE.DoubleSide,
                shininess: 20,
                transparent: false
            });
            
            return new THREE.Mesh(geometry, material);
        }
        
        function createBoundaryLines() {
            if (!densityData.boundaries || densityData.boundaries.length === 0) return null;
            
            const group = new THREE.Group();
            const material = new THREE.LineBasicMaterial({ 
                color: 0x88ccff,  // Brighter blue for visibility
                transparent: true,
                opacity: 0.8,
                linewidth: 1
            });
            
            const boundaryRadius = GLOBE_RADIUS * 1.005; // Slightly above surface
            
            for (const boundary of densityData.boundaries) {
                if (boundary.length < 2) continue;
                
                const points = [];
                for (const [lon, lat] of boundary) {
                    const pos = latLonToVector3(lat, lon, boundaryRadius);
                    points.push(pos);
                }
                
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const line = new THREE.Line(geometry, material);
                group.add(line);
            }
            
            return group;
        }
        
        function updateVisualization() {
            const resKey = settings.resolution.toString();
            if (!densityData.histograms[resKey]) {
                // Find closest available resolution
                const available = Object.keys(densityData.histograms).map(Number).sort((a, b) => a - b);
                const closest = available.reduce((a, b) => 
                    Math.abs(b - settings.resolution) < Math.abs(a - settings.resolution) ? b : a
                );
                settings.resolution = closest;
            }
            
            const processed = processHistogram(
                densityData.histograms[settings.resolution.toString()],
                settings.sigma,
                settings.power
            );
            
            if (densityMesh) {
                scene.remove(densityMesh);
                densityMesh.geometry.dispose();
                densityMesh.material.dispose();
            }
            
            densityMesh = createDensityMesh(processed);
            scene.add(densityMesh);
            
            if (boundaryLines) {
                boundaryLines.visible = settings.showBoundaries;
            }
        }
        
        function focusOnUSA() {
            // Animate camera to focus on continental US
            const targetLat = 39;
            const targetLon = -98;
            const pos = latLonToVector3(targetLat, targetLon, GLOBE_RADIUS * 2.5);
            
            // Simple animation
            const startPos = camera.position.clone();
            const duration = 1000;
            const startTime = Date.now();
            
            function animateCamera() {
                const elapsed = Date.now() - startTime;
                const t = Math.min(1, elapsed / duration);
                const eased = t * t * (3 - 2 * t); // Smooth step
                
                camera.position.lerpVectors(startPos, pos, eased);
                camera.lookAt(0, 0, 0);
                
                if (t < 1) {
                    requestAnimationFrame(animateCamera);
                }
            }
            animateCamera();
        }
        
        function sideView() {
            // Animate camera to a horizontal side view of the globe
            const pos = new THREE.Vector3(GLOBE_RADIUS * 3, 0, 0);
            
            const startPos = camera.position.clone();
            const startTarget = controls.target.clone();
            const endTarget = new THREE.Vector3(0, 0, 0);
            const duration = 1000;
            const startTime = Date.now();
            
            function animateCamera() {
                const elapsed = Date.now() - startTime;
                const t = Math.min(1, elapsed / duration);
                const eased = t * t * (3 - 2 * t);
                
                camera.position.lerpVectors(startPos, pos, eased);
                controls.target.lerpVectors(startTarget, endTarget, eased);
                
                if (t < 1) {
                    requestAnimationFrame(animateCamera);
                }
            }
            animateCamera();
        }
        
        function resetView() {
            // Reset camera to default position
            const pos = new THREE.Vector3(0, 0, 15);
            
            const startPos = camera.position.clone();
            const startTarget = controls.target.clone();
            const endTarget = new THREE.Vector3(0, 0, 0);
            const duration = 800;
            const startTime = Date.now();
            
            function animateCamera() {
                const elapsed = Date.now() - startTime;
                const t = Math.min(1, elapsed / duration);
                const eased = t * t * (3 - 2 * t);
                
                camera.position.lerpVectors(startPos, pos, eased);
                controls.target.lerpVectors(startTarget, endTarget, eased);
                
                if (t < 1) {
                    requestAnimationFrame(animateCamera);
                }
            }
            animateCamera();
        }
        
        function initScene() {
            const container = document.getElementById('canvas-container');
            
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a15);
            
            // Add stars
            const starsGeometry = new THREE.BufferGeometry();
            const starPositions = [];
            for (let i = 0; i < 2000; i++) {
                const r = 100;
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.acos(2 * Math.random() - 1);
                starPositions.push(
                    r * Math.sin(phi) * Math.cos(theta),
                    r * Math.sin(phi) * Math.sin(theta),
                    r * Math.cos(phi)
                );
            }
            starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3));
            const starsMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.3 });
            scene.add(new THREE.Points(starsGeometry, starsMaterial));
            
            const aspect = window.innerWidth / window.innerHeight;
            camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
            camera.position.set(0, 0, 15);
            camera.lookAt(0, 0, 0);
            
            renderer = new THREE.WebGLRenderer({ 
                antialias: true,
                alpha: false,
                powerPreference: 'high-performance',
                failIfMajorPerformanceCaveat: false,  // Allow software rendering fallback
                preserveDrawingBuffer: true  // Better compatibility with some browsers
            });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);
            
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;       // Smoother damping
            controls.minDistance = GLOBE_RADIUS * 1.2;  // Allow closer zoom
            controls.maxDistance = GLOBE_RADIUS * 15;   // Allow further zoom out
            controls.autoRotate = settings.autoRotate;  // Use default from settings
            controls.autoRotateSpeed = 0.5;
            // Allow full rotation - no angle restrictions
            controls.minPolarAngle = 0;           // Can look from top
            controls.maxPolarAngle = Math.PI;     // Can look from bottom
            controls.enablePan = true;            // Allow panning
            controls.panSpeed = 0.8;
            controls.rotateSpeed = 0.6;
            controls.zoomSpeed = 0.8;             // Smoother zoom
            controls.enableZoom = true;
            // Smooth zoom interpolation
            controls.zoomToCursor = false;
            
            // Lighting
            scene.add(new THREE.AmbientLight(0xffffff, 0.3));
            
            const sunLight = new THREE.DirectionalLight(0xffffff, 1.0);
            sunLight.position.set(10, 5, 10);
            scene.add(sunLight);
            
            const fillLight = new THREE.DirectionalLight(0x4466aa, 0.3);
            fillLight.position.set(-5, -5, -5);
            scene.add(fillLight);
            
            // Create globe with default base map
            globeMesh = createGlobe(settings.baseMap);
            scene.add(globeMesh);
            
            atmosphereMesh = createAtmosphere();
            scene.add(atmosphereMesh);
            
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
            
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        }
        
        function setupControls() {
            const resSelect = document.getElementById('resolution');
            const resLabels = {
                '90': 'Very Low (90) - 2°',
                '180': 'Low (180) - 1°',
                '360': 'Medium (360) - 0.5°',
                '720': 'High (720) - 0.25°',
                '1080': 'Very High (1080) - 0.17°',
                '1440': 'Ultra (1440) - 0.125°',
                '2880': 'Extreme (2880) - 0.0625°',
                '3600': 'Maximum (3600) - 0.05°'
            };
            
            resSelect.innerHTML = '';
            const sortedResolutions = Object.keys(densityData.histograms).sort((a, b) => +a - +b);
            for (const res of sortedResolutions) {
                const opt = document.createElement('option');
                opt.value = res;
                opt.textContent = resLabels[res] || `${res} bins`;
                resSelect.appendChild(opt);
            }
            
            // Default to highest resolution
            const defaultRes = sortedResolutions[sortedResolutions.length - 1];
            settings.resolution = parseInt(defaultRes);
            resSelect.value = defaultRes;
            
            resSelect.addEventListener('change', (e) => {
                settings.resolution = parseInt(e.target.value);
                updateVisualization();
            });
            
            const sigmaSlider = document.getElementById('sigma');
            const sigmaValue = document.getElementById('sigma-value');
            sigmaSlider.addEventListener('input', (e) => {
                settings.sigma = parseFloat(e.target.value);
                sigmaValue.textContent = settings.sigma.toFixed(2);
            });
            sigmaSlider.addEventListener('change', () => updateVisualization());
            
            const powerSlider = document.getElementById('power');
            const powerValue = document.getElementById('power-value');
            powerSlider.addEventListener('input', (e) => {
                settings.power = parseFloat(e.target.value);
                powerValue.textContent = settings.power.toFixed(1);
            });
            powerSlider.addEventListener('change', () => updateVisualization());
            
            const heightSlider = document.getElementById('height-scale');
            const heightValue = document.getElementById('height-scale-value');
            heightSlider.addEventListener('input', (e) => {
                settings.heightScale = parseFloat(e.target.value);
                heightValue.textContent = settings.heightScale.toFixed(2);
            });
            heightSlider.addEventListener('change', () => updateVisualization());
            
            const thresholdSlider = document.getElementById('threshold');
            const thresholdValue = document.getElementById('threshold-value');
            thresholdSlider.addEventListener('input', (e) => {
                settings.threshold = parseFloat(e.target.value);
                thresholdValue.textContent = settings.threshold.toFixed(3);
            });
            thresholdSlider.addEventListener('change', () => updateVisualization());
            
            document.getElementById('show-boundaries').addEventListener('change', (e) => {
                settings.showBoundaries = e.target.value === 'true';
                if (boundaryLines) boundaryLines.visible = settings.showBoundaries;
            });
            
            document.getElementById('auto-rotate').addEventListener('change', (e) => {
                settings.autoRotate = e.target.value === 'true';
                controls.autoRotate = settings.autoRotate;
            });
            
            document.getElementById('base-map').addEventListener('change', (e) => {
                settings.baseMap = e.target.value;
                updateGlobeBaseMap(settings.baseMap);
            });
            
            document.getElementById('focus-btn').addEventListener('click', focusOnUSA);
            document.getElementById('side-view-btn').addEventListener('click', sideView);
            document.getElementById('reset-view-btn').addEventListener('click', resetView);
            
            document.getElementById('point-count').textContent = 
                densityData.total_points.toLocaleString();
            
            // Setup peaks panel
            setupPeaksPanel();
            
            // Setup navigation widget
            setupNavWidget();
        }
        
        // ============================================
        // NAVIGATION WIDGET
        // ============================================
        
        function setupNavWidget() {
            document.getElementById('nav-widget').classList.remove('hidden');
            
            // Smooth animated camera movement
            let isAnimating = false;
            
            function smoothCameraMove(deltaAzimuth, deltaPolar, deltaZoom, duration = 300) {
                if (isAnimating) return;
                
                const startAzimuth = controls.getAzimuthalAngle();
                const startPolar = controls.getPolarAngle();
                const startDistance = camera.position.length();
                
                const targetAzimuth = startAzimuth + deltaAzimuth;
                const targetPolar = Math.max(0.1, Math.min(Math.PI - 0.1, startPolar + deltaPolar));
                const targetDistance = Math.max(controls.minDistance, Math.min(controls.maxDistance, startDistance * (1 + deltaZoom)));
                
                isAnimating = true;
                const startTime = Date.now();
                
                function animate() {
                    const elapsed = Date.now() - startTime;
                    const t = Math.min(1, elapsed / duration);
                    // Ease out cubic for smooth deceleration
                    const eased = 1 - Math.pow(1 - t, 3);
                    
                    const currentAzimuth = startAzimuth + (targetAzimuth - startAzimuth) * eased;
                    const currentPolar = startPolar + (targetPolar - startPolar) * eased;
                    const currentDistance = startDistance + (targetDistance - startDistance) * eased;
                    
                    // Convert spherical to cartesian
                    camera.position.x = currentDistance * Math.sin(currentPolar) * Math.cos(currentAzimuth);
                    camera.position.y = currentDistance * Math.cos(currentPolar);
                    camera.position.z = currentDistance * Math.sin(currentPolar) * Math.sin(currentAzimuth);
                    
                    camera.lookAt(controls.target);
                    
                    if (t < 1) {
                        requestAnimationFrame(animate);
                    } else {
                        isAnimating = false;
                    }
                }
                animate();
            }
            
            // Rotation buttons
            const rotateStep = Math.PI / 12; // 15 degrees
            
            document.getElementById('nav-left').addEventListener('click', () => {
                smoothCameraMove(-rotateStep, 0, 0);
            });
            
            document.getElementById('nav-right').addEventListener('click', () => {
                smoothCameraMove(rotateStep, 0, 0);
            });
            
            document.getElementById('nav-up').addEventListener('click', () => {
                smoothCameraMove(0, -rotateStep, 0);
            });
            
            document.getElementById('nav-down').addEventListener('click', () => {
                smoothCameraMove(0, rotateStep, 0);
            });
            
            // Reset view
            document.getElementById('nav-reset').addEventListener('click', resetView);
            
            // Zoom buttons with smooth animation
            document.getElementById('zoom-in').addEventListener('click', () => {
                smoothCameraMove(0, 0, -0.2, 250);
            });
            
            document.getElementById('zoom-out').addEventListener('click', () => {
                smoothCameraMove(0, 0, 0.25, 250);
            });
            
            // Tilt buttons (polar angle)
            const tiltStep = Math.PI / 18; // 10 degrees
            
            document.getElementById('tilt-up').addEventListener('click', () => {
                smoothCameraMove(0, -tiltStep, 0, 200);
            });
            
            document.getElementById('tilt-down').addEventListener('click', () => {
                smoothCameraMove(0, tiltStep, 0, 200);
            });
            
            // Hold-to-repeat for navigation buttons
            const repeatButtons = ['nav-up', 'nav-down', 'nav-left', 'nav-right', 'zoom-in', 'zoom-out', 'tilt-up', 'tilt-down'];
            repeatButtons.forEach(btnId => {
                const btn = document.getElementById(btnId);
                let intervalId = null;
                
                btn.addEventListener('mousedown', () => {
                    intervalId = setInterval(() => btn.click(), 150);
                });
                
                btn.addEventListener('mouseup', () => clearInterval(intervalId));
                btn.addEventListener('mouseleave', () => clearInterval(intervalId));
            });
        }
        
        // ============================================
        // PEAKS PANEL AND LOCAL VIEW FUNCTIONALITY
        // ============================================
        
        let localScene, localCamera, localRenderer, localControls;
        let localDensityMesh, localMap;
        let currentPeakIndex = -1;
        let currentLocalData = null;  // Store current local data globally
        let localAnimationId = null;  // Track animation frame
        
        const localSettings = {
            sigma: 0.05,
            power: 2.0,
            heightScale: 0.5,
            threshold: 0.01,
            showMapOverlay: false
        };
        
        function setupPeaksPanel() {
            if (!densityData.peaks || densityData.peaks.length === 0) {
                return;
            }
            
            const peaksList = document.getElementById('peaks-list');
            peaksList.innerHTML = '';
            
            densityData.peaks.forEach((peak, index) => {
                const item = document.createElement('div');
                item.className = 'peak-item';
                item.innerHTML = `
                    <div class="peak-rank">${index + 1}</div>
                    <div class="peak-info">
                        <div>${peak.count.toLocaleString()} obs</div>
                        <div class="coords">${peak.lat.toFixed(2)}°, ${peak.lon.toFixed(2)}°</div>
                    </div>
                `;
                item.addEventListener('click', () => openLocalView(index));
                peaksList.appendChild(item);
            });
            
            document.getElementById('peaks-panel').classList.remove('hidden');
        }
        
        function openLocalView(peakIndex) {
            // Clean up any existing local view first
            if (currentPeakIndex !== -1) {
                cleanupLocalView();
            }
            
            currentPeakIndex = peakIndex;
            const peak = densityData.peaks[peakIndex];
            const localData = densityData.local_views[peakIndex.toString()];
            
            if (!localData) {
                console.error('No local data for peak', peakIndex);
                return;
            }
            
            // Store globally for slider updates
            currentLocalData = localData;
            
            // Update title
            document.getElementById('local-title').textContent = `Peak #${peakIndex + 1}`;
            document.getElementById('local-subtitle').textContent = 
                `${(localData.radius_km * 2).toFixed(0)}km × ${(localData.radius_km * 2).toFixed(0)}km region • ${localData.total_points.toLocaleString()} points`;
            
            // Show overlay
            document.getElementById('local-view-overlay').classList.add('active');
            
            // Initialize map and 3D view
            setTimeout(() => {
                initLocalMap(localData);
                initLocal3D(localData);
                setupLocalControls();  // No argument needed now
            }, 100);
        }
        
        function cleanupLocalView() {
            // Stop animation
            if (localAnimationId) {
                cancelAnimationFrame(localAnimationId);
                localAnimationId = null;
            }
            
            // Clean up map
            if (localMap) {
                localMap.remove();
                localMap = null;
            }
            
            // Clean up 3D
            if (localDensityMesh) {
                disposeObject(localDensityMesh);
                localDensityMesh = null;
            }
            if (localScene) {
                // Dispose all children
                while(localScene.children.length > 0) {
                    disposeObject(localScene.children[0]);
                    localScene.remove(localScene.children[0]);
                }
            }
            if (localRenderer) {
                localRenderer.dispose();
                localRenderer = null;
            }
            if (localControls) {
                localControls.dispose();
                localControls = null;
            }
            
            const canvas = document.getElementById('local-canvas');
            if (canvas) canvas.innerHTML = '';
        }
        
        // Helper to recursively dispose Three.js objects
        function disposeObject(obj) {
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) {
                if (Array.isArray(obj.material)) {
                    obj.material.forEach(m => m.dispose());
                } else {
                    obj.material.dispose();
                }
            }
            if (obj.children) {
                obj.children.forEach(child => disposeObject(child));
            }
        }
        
        function closeLocalView() {
            document.getElementById('local-view-overlay').classList.remove('active');
            cleanupLocalView();
            currentPeakIndex = -1;
            currentLocalData = null;
        }
        
        function initLocalMap(localData) {
            const bounds = localData.bounds;
            const center = localData.center;
            
            // Remove existing map if any
            if (localMap) {
                localMap.remove();
            }
            
            // Create map
            localMap = L.map('local-map', {
                center: [center.lat, center.lon],
                zoom: 11,
                zoomControl: true
            });
            
            // Add OpenStreetMap tiles (Carto light style for cleaner look)
            L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
                attribution: '© OpenStreetMap contributors © CARTO',
                subdomains: 'abcd',
                maxZoom: 19
            }).addTo(localMap);
            
            // Fit to bounds
            localMap.fitBounds([
                [bounds.lat_min, bounds.lon_min],
                [bounds.lat_max, bounds.lon_max]
            ]);
            
            // Add rectangle showing the region
            L.rectangle([
                [bounds.lat_min, bounds.lon_min],
                [bounds.lat_max, bounds.lon_max]
            ], {
                color: '#e94560',
                weight: 2,
                fillOpacity: 0.1
            }).addTo(localMap);
            
            // Add center marker
            L.circleMarker([center.lat, center.lon], {
                radius: 8,
                color: '#e94560',
                fillColor: '#e94560',
                fillOpacity: 0.8
            }).addTo(localMap);
        }
        
        function initLocal3D(localData) {
            const container = document.getElementById('local-canvas');
            container.innerHTML = '';
            
            const width = container.clientWidth || window.innerWidth / 2;
            const height = container.clientHeight || window.innerHeight;
            
            // Scene
            localScene = new THREE.Scene();
            localScene.background = new THREE.Color(0x1a1a2e);
            
            // Camera - angled view
            localCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            localCamera.position.set(0, -8, 6);
            localCamera.lookAt(0, 0, 0);
            
            // Renderer
            localRenderer = new THREE.WebGLRenderer({ 
                antialias: true,
                alpha: false,
                powerPreference: 'high-performance',
                failIfMajorPerformanceCaveat: false,
                preserveDrawingBuffer: true
            });
            localRenderer.setSize(width, height);
            localRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(localRenderer.domElement);
            
            // Controls
            localControls = new THREE.OrbitControls(localCamera, localRenderer.domElement);
            localControls.enableDamping = true;
            localControls.dampingFactor = 0.05;
            localControls.minDistance = 3;
            localControls.maxDistance = 20;
            
            // Lighting
            localScene.add(new THREE.AmbientLight(0xffffff, 0.4));
            
            const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
            light1.position.set(5, 5, 10);
            localScene.add(light1);
            
            const light2 = new THREE.DirectionalLight(0xffffff, 0.3);
            light2.position.set(-5, -5, 5);
            localScene.add(light2);
            
            // Create density visualization
            updateLocalVisualization();
            
            // Animation loop
            function animateLocal() {
                if (currentPeakIndex === -1 || !localRenderer) return;
                localAnimationId = requestAnimationFrame(animateLocal);
                if (localControls) localControls.update();
                if (localRenderer && localScene && localCamera) {
                    localRenderer.render(localScene, localCamera);
                }
            }
            animateLocal();
        }
        
        function updateLocalVisualization() {
            if (!currentLocalData || !localScene) return;
            
            // Process histogram
            const histData = currentLocalData.histogram;
            const [width, height] = histData.shape;
            
            // Decompress
            const data = new Float32Array(width * height);
            for (let i = 0; i < histData.indices.length; i++) {
                const [x, y] = histData.indices[i];
                data[y * width + x] = histData.values[i];
            }
            
            // Apply gaussian blur
            const blurred = gaussianBlur(data, width, height, localSettings.sigma);
            
            // Apply power transformation
            const result = new Float32Array(width * height);
            let maxVal = 0;
            for (let i = 0; i < blurred.length; i++) {
                if (blurred[i] > 0) {
                    result[i] = Math.pow(Math.log1p(blurred[i]), localSettings.power);
                    maxVal = Math.max(maxVal, result[i]);
                }
            }
            if (maxVal > 0) {
                for (let i = 0; i < result.length; i++) {
                    result[i] /= maxVal;
                }
            }
            
            // Remove old mesh
            if (localDensityMesh) {
                localScene.remove(localDensityMesh);
                disposeObject(localDensityMesh);
                localDensityMesh = null;
            }
            
            // Create mesh
            localDensityMesh = createLocalDensityMesh(result, width, height, currentLocalData.bounds);
            localScene.add(localDensityMesh);
        }
        
        function createLocalDensityMesh(data, width, height, bounds) {
            const geometry = new THREE.BufferGeometry();
            const vertices = [];
            const colors = [];
            const indices = [];
            
            const scaleX = 8;
            const scaleY = 8 * ((bounds.lat_max - bounds.lat_min) / (bounds.lon_max - bounds.lon_min));
            const heightScale = localSettings.heightScale * 4;
            const threshold = localSettings.threshold;
            
            let vertexCount = 0;
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    const z = data[idx] || 0;
                    
                    if (z < threshold) continue;
                    
                    // Grid position to XY
                    const px = (x + 0.5) / width - 0.5;
                    const py = (y + 0.5) / height - 0.5;
                    
                    // Cell size
                    const dx = 1 / width;
                    const dy = 1 / height;
                    
                    // Base and peak heights
                    const baseZ = 0;
                    const peakZ = z * heightScale;
                    
                    // 4 corners
                    const x0 = (px - dx/2) * scaleX, x1 = (px + dx/2) * scaleX;
                    const y0 = (py - dy/2) * scaleY, y1 = (py + dy/2) * scaleY;
                    
                    // Colors
                    const color = interpolateColor(z);
                    const r = color[0]/255, g = color[1]/255, b = color[2]/255;
                    const rd = r*0.5, gd = g*0.5, bd = b*0.5;
                    
                    const baseIdx = vertexCount;
                    
                    // 8 vertices: 4 base + 4 peak
                    vertices.push(x0, y0, baseZ); colors.push(rd, gd, bd);
                    vertices.push(x1, y0, baseZ); colors.push(rd, gd, bd);
                    vertices.push(x1, y1, baseZ); colors.push(rd, gd, bd);
                    vertices.push(x0, y1, baseZ); colors.push(rd, gd, bd);
                    vertices.push(x0, y0, peakZ); colors.push(r, g, b);
                    vertices.push(x1, y0, peakZ); colors.push(r, g, b);
                    vertices.push(x1, y1, peakZ); colors.push(r, g, b);
                    vertices.push(x0, y1, peakZ); colors.push(r, g, b);
                    
                    // Top face
                    indices.push(baseIdx+4, baseIdx+5, baseIdx+6);
                    indices.push(baseIdx+4, baseIdx+6, baseIdx+7);
                    
                    // Sides
                    indices.push(baseIdx+0, baseIdx+4, baseIdx+1);
                    indices.push(baseIdx+1, baseIdx+4, baseIdx+5);
                    indices.push(baseIdx+1, baseIdx+5, baseIdx+2);
                    indices.push(baseIdx+2, baseIdx+5, baseIdx+6);
                    indices.push(baseIdx+2, baseIdx+6, baseIdx+3);
                    indices.push(baseIdx+3, baseIdx+6, baseIdx+7);
                    indices.push(baseIdx+3, baseIdx+7, baseIdx+0);
                    indices.push(baseIdx+0, baseIdx+7, baseIdx+4);
                    
                    vertexCount += 8;
                }
            }
            
            if (vertices.length === 0) {
                return new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            geometry.setIndex(indices);
            geometry.computeVertexNormals();
            
            // Add a base plane (with optional map texture)
            const planeGeo = new THREE.PlaneGeometry(scaleX * 1.05, scaleY * 1.05);
            let planeMat;
            
            if (localSettings.showMapOverlay && currentLocalData) {
                // Create a canvas to render map tiles
                const mapCanvas = document.createElement('canvas');
                const canvasSize = 2048;  // High resolution canvas for detailed maps
                mapCanvas.width = canvasSize;
                mapCanvas.height = canvasSize;
                const ctx = mapCanvas.getContext('2d');
                
                // Fill with base color first
                ctx.fillStyle = '#2a3a5a';
                ctx.fillRect(0, 0, canvasSize, canvasSize);
                
                // Get data bounds
                const b = currentLocalData.bounds;
                
                // Calculate appropriate zoom level based on region size
                const latSpan = b.lat_max - b.lat_min;
                const lonSpan = b.lon_max - b.lon_min;
                const maxSpan = Math.max(latSpan, lonSpan);
                // Zoom level where 1 tile ≈ 360/2^zoom degrees
                // Higher zoom = more detail. For ~100km regions, zoom 13-16 is ideal
                const zoom = Math.max(12, Math.min(18, Math.floor(Math.log2(360 / maxSpan)) + 1));
                
                // Calculate tile bounds
                const n = Math.pow(2, zoom);
                const minTileX = Math.floor((b.lon_min + 180) / 360 * n);
                const maxTileX = Math.floor((b.lon_max + 180) / 360 * n);
                const minTileY = Math.floor((1 - Math.log(Math.tan(b.lat_max * Math.PI / 180) + 1 / Math.cos(b.lat_max * Math.PI / 180)) / Math.PI) / 2 * n);
                const maxTileY = Math.floor((1 - Math.log(Math.tan(b.lat_min * Math.PI / 180) + 1 / Math.cos(b.lat_min * Math.PI / 180)) / Math.PI) / 2 * n);
                
                // Create texture from canvas
                const texture = new THREE.CanvasTexture(mapCanvas);
                texture.needsUpdate = true;
                
                // Calculate tile extent in lat/lon
                const tile2lon = (x, z) => x / Math.pow(2, z) * 360 - 180;
                const tile2lat = (y, z) => {
                    const n = Math.PI - 2 * Math.PI * y / Math.pow(2, z);
                    return 180 / Math.PI * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
                };
                
                // Load tiles and draw them correctly positioned
                const numTilesX = maxTileX - minTileX + 1;
                const numTilesY = maxTileY - minTileY + 1;
                const tileSize = canvasSize / Math.max(numTilesX, numTilesY);
                
                let tilesLoaded = 0;
                const totalTiles = numTilesX * numTilesY;
                
                for (let tx = minTileX; tx <= maxTileX; tx++) {
                    for (let ty = minTileY; ty <= maxTileY; ty++) {
                        const tileImg = new Image();
                        tileImg.crossOrigin = 'anonymous';
                        
                        // Calculate tile bounds
                        const tileLonMin = tile2lon(tx, zoom);
                        const tileLonMax = tile2lon(tx + 1, zoom);
                        const tileLatMax = tile2lat(ty, zoom);
                        const tileLatMin = tile2lat(ty + 1, zoom);
                        
                        // Calculate position in canvas (normalized to data bounds)
                        const canvasX = (tileLonMin - b.lon_min) / lonSpan * canvasSize;
                        const canvasY = (b.lat_max - tileLatMax) / latSpan * canvasSize;
                        const canvasW = (tileLonMax - tileLonMin) / lonSpan * canvasSize;
                        const canvasH = (tileLatMax - tileLatMin) / latSpan * canvasSize;
                        
                        tileImg.onload = () => {
                            ctx.drawImage(tileImg, canvasX, canvasY, canvasW, canvasH);
                            tilesLoaded++;
                            texture.needsUpdate = true;
                        };
                        tileImg.onerror = () => {
                            tilesLoaded++;
                        };
                        tileImg.src = `https://tile.openstreetmap.org/${zoom}/${tx}/${ty}.png`;
                    }
                }
                
                planeMat = new THREE.MeshBasicMaterial({ 
                    map: texture,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.8
                });
            } else {
                planeMat = new THREE.MeshBasicMaterial({ color: 0x2a3a5a, side: THREE.DoubleSide });
            }
            
            const plane = new THREE.Mesh(planeGeo, planeMat);
            plane.position.z = -0.01;
            
            const group = new THREE.Group();
            group.add(plane);
            
            const material = new THREE.MeshPhongMaterial({
                vertexColors: true,
                side: THREE.DoubleSide,
                shininess: 20
            });
            const mesh = new THREE.Mesh(geometry, material);
            group.add(mesh);
            
            return group;
        }
        
        function setupLocalControls() {
            const sigmaSlider = document.getElementById('local-sigma');
            const sigmaValue = document.getElementById('local-sigma-value');
            sigmaSlider.value = localSettings.sigma;
            sigmaValue.textContent = localSettings.sigma.toFixed(2);
            
            sigmaSlider.oninput = (e) => {
                localSettings.sigma = parseFloat(e.target.value);
                sigmaValue.textContent = localSettings.sigma.toFixed(2);
            };
            sigmaSlider.onchange = () => updateLocalVisualization();
            
            const powerSlider = document.getElementById('local-power');
            const powerValue = document.getElementById('local-power-value');
            powerSlider.value = localSettings.power;
            powerValue.textContent = localSettings.power.toFixed(1);
            
            powerSlider.oninput = (e) => {
                localSettings.power = parseFloat(e.target.value);
                powerValue.textContent = localSettings.power.toFixed(1);
            };
            powerSlider.onchange = () => updateLocalVisualization();
            
            const heightSlider = document.getElementById('local-height');
            const heightValue = document.getElementById('local-height-value');
            heightSlider.value = localSettings.heightScale;
            heightValue.textContent = localSettings.heightScale.toFixed(2);
            
            heightSlider.oninput = (e) => {
                localSettings.heightScale = parseFloat(e.target.value);
                heightValue.textContent = localSettings.heightScale.toFixed(2);
            };
            heightSlider.onchange = () => updateLocalVisualization();
            
            const thresholdSlider = document.getElementById('local-threshold');
            const thresholdValue = document.getElementById('local-threshold-value');
            thresholdSlider.value = localSettings.threshold;
            thresholdValue.textContent = localSettings.threshold.toFixed(3);
            
            thresholdSlider.oninput = (e) => {
                localSettings.threshold = parseFloat(e.target.value);
                thresholdValue.textContent = localSettings.threshold.toFixed(3);
            };
            thresholdSlider.onchange = () => updateLocalVisualization();
            
            // Map overlay toggle
            document.getElementById('local-map-overlay').addEventListener('change', (e) => {
                localSettings.showMapOverlay = e.target.value === 'true';
                updateLocalVisualization();
            });
            
            // Close button
            document.getElementById('close-local-btn').onclick = closeLocalView;
        }
        
        async function init() {
            // Check WebGL availability
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            if (!gl) {
                document.getElementById('loading').innerHTML = 
                    `<p style="color: #e94560;">WebGL is not supported in this browser.<br>
                    Please try Safari, Chrome, or Firefox.</p>`;
                return;
            }
            
            try {
                initScene();
            } catch (e) {
                console.error('Failed to initialize 3D scene:', e);
                document.getElementById('loading').innerHTML = 
                    `<p style="color: #e94560;">Failed to initialize 3D graphics.<br>
                    Error: ${e.message}</p>`;
                return;
            }
            
            try {
                const compressed = Uint8Array.from(atob(EMBEDDED_DATA), c => c.charCodeAt(0));
                const jsonText = pako.ungzip(compressed, { to: 'string' });
                densityData = JSON.parse(jsonText);
                
                console.log('Loaded globe data:', {
                    resolutions: Object.keys(densityData.histograms),
                    totalPoints: densityData.total_points,
                    boundaries: densityData.boundaries?.length || 0,
                    peaks: densityData.peaks?.length || 0,
                    localViews: Object.keys(densityData.local_views || {}).length
                });
                
                boundaryLines = createBoundaryLines();
                if (boundaryLines) scene.add(boundaryLines);
                
                setupControls();
                updateVisualization();
                
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('controls').classList.remove('hidden');
                document.getElementById('stats').classList.remove('hidden');
                document.getElementById('colorbar').classList.remove('hidden');
                document.getElementById('colorbar-labels').classList.remove('hidden');
                
                // Show peaks panel if peaks exist
                if (densityData.peaks && densityData.peaks.length > 0) {
                    document.getElementById('peaks-panel').classList.remove('hidden');
                }
                
                // Start focused on USA
                setTimeout(focusOnUSA, 500);
                
            } catch (error) {
                console.error('Error loading data:', error);
                document.getElementById('loading').innerHTML = 
                    `<p style="color: #e94560;">Error loading data: ${error.message}</p>`;
            }
        }
        
        init();
    </script>
</body>
</html>'''

    return html_template


def main():
    parser = argparse.ArgumentParser(description='Generate static HTML globe density viewer')
    parser.add_argument('-i', '--input', default='output/raw.tsv.gz',
                        help='Input TSV file')
    parser.add_argument('-o', '--output', default='output/viewer/globe_viewer.html',
                        help='Output HTML file')
    parser.add_argument('--resolutions', type=int, nargs='+', default=DEFAULT_RESOLUTIONS,
                        help=f'Resolutions to compute (default: {DEFAULT_RESOLUTIONS}). '
                             'Higher = more detail. 360 = 0.5° bins')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Rows per chunk (default: {DEFAULT_CHUNK_SIZE:,})')
    parser.add_argument('--shapefile', default='output/cb_2020_us_county_500k.shp',
                        help='Shapefile for US state/county boundaries')
    parser.add_argument('--sigma', type=float, default=0.0,
                        help='Default smoothing sigma (0-0.5, 0 = sharp peaks, higher = smoother)')
    parser.add_argument('--power', type=float, default=2.0,
                        help='Default power exponent')
    parser.add_argument('--n-peaks', type=int, default=10,
                        help='Number of top peaks to include local views for')
    parser.add_argument('--local-radius', type=float, default=50.0,
                        help='Radius in km for local peak views (default: 50 = 100km x 100km)')
    parser.add_argument('--local-resolution', type=int, default=200,
                        help='Resolution for local peak histograms (default: 200)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for local histograms (default: 4)')
    args = parser.parse_args()

    density_data = {
        'bounds': {
            'lat_min': WORLD_LAT_MIN,
            'lat_max': WORLD_LAT_MAX,
            'lon_min': WORLD_LON_MIN,
            'lon_max': WORLD_LON_MAX
        },
        'histograms': {},
        'boundaries': [],
        'peaks': [],
        'local_views': {},
        'total_points': 0
    }

    # Compute histograms
    print(f"\n📊 Computing global histograms at {len(args.resolutions)} resolutions...")
    histograms, total_points = build_histograms_multi_resolution(
        args.input, args.resolutions, args.chunk_size
    )
    density_data['total_points'] = total_points

    for res in sorted(args.resolutions):
        hist = histograms[res]
        density_data['histograms'][str(res)] = compress_histogram(hist)
        nonzero = (hist > 0).sum()
        lat_bins = res // 2
        print(f"   ✓ {res} x {lat_bins} (lon x lat): {nonzero:,} non-zero cells")

    # Find top peaks
    print(f"\n🏔️  Finding top {args.n_peaks} peaks...")
    peaks = find_top_peaks(histograms, n_peaks=args.n_peaks)
    density_data['peaks'] = peaks
    for i, peak in enumerate(peaks):
        print(f"   {i+1}. ({peak['lat']:.2f}, {peak['lon']:.2f}) - {peak['count']:,} observations")

    # Build local high-resolution histograms for each peak (parallel)
    print(f"\n🔍 Building local views ({args.local_radius*2:.0f}km × {args.local_radius*2:.0f}km regions) using {args.workers} workers...")
    density_data['local_views'] = build_local_histograms_parallel(
        args.input,
        peaks,
        radius_km=args.local_radius,
        resolution=args.local_resolution,
        chunk_size=args.chunk_size,
        max_workers=args.workers
    )

    # Extract boundaries (US states + world countries if available)
    print(f"\n🗺️  Extracting boundaries...")
    boundaries = []

    # World boundaries first (if available)
    world_boundaries = get_world_boundaries()
    if world_boundaries:
        boundaries.extend(world_boundaries)
        print(f"   ✓ {len(world_boundaries)} world boundary polygons")

    # US state/county boundaries
    us_boundaries = extract_us_boundaries(args.shapefile, simplify_factor=80)
    if us_boundaries:
        boundaries.extend(us_boundaries)
        print(f"   ✓ {len(us_boundaries)} US boundary polygons")

    density_data['boundaries'] = boundaries
    print(f"   Total: {len(boundaries)} boundary polygons")

    # Generate HTML
    print(f"\n📄 Generating static HTML globe viewer...")
    html_content = generate_html(density_data, args.sigma, args.power)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Output: {args.output} ({size_mb:.2f} MB)")

    print(f"\n🎉 Done! Open the HTML file directly in a browser:")
    print(f"   open {args.output}")


if __name__ == '__main__':
    main()
