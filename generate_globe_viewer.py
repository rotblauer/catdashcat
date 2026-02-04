#!/usr/bin/env python3
"""Generate a self-contained static HTML globe viewer with embedded density data.

This creates a single HTML file with an interactive 3D globe that can be opened
directly in a browser without needing a web server.

Example usage:
    python generate_globe_viewer.py -i output/raw.tsv.gz -o output/viewer/globe_viewer.html
    python generate_globe_viewer.py --resolutions 100 250 500 --sigma 1.2 --power 2.5
    .venv/bin/python generate_globe_viewer.py --resolutions 180 360 720
    .venv/bin/python generate_globe_viewer.py --resolutions 720 1440 2880 --sigma 0.05

"""

import argparse
import base64
import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path

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
        
        #focus-btn {
            width: 100%;
            padding: 8px;
            margin-top: 10px;
            background: #e94560;
            border: none;
            border-radius: 6px;
            color: white;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        #focus-btn:hover { background: #ff6b6b; }
    </style>
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
                <input type="range" id="sigma" min="0.1" max="4" step="0.1" value="''' + str(default_sigma) + '''">
                <div class="value" id="sigma-value">''' + str(default_sigma) + '''</div>
                <div class="hint">Lower = sharper peaks</div>
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
                    <option value="false" selected>Off</option>
                    <option value="true">On</option>
                </select>
            </div>
            
            <button id="focus-btn">🇺🇸 Focus on USA</button>
        </div>
        
        <div id="stats" class="hidden">
            GPS observations: <span id="point-count">0</span> |
            Drag to rotate, scroll to zoom
        </div>
        
        <div id="colorbar" class="hidden"></div>
        <div id="colorbar-labels" class="hidden">
            <span>High</span>
            <span>Low</span>
        </div>
    </div>

    <!-- Three.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.134.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
    
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
            autoRotate: false
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
            if (sigma < 0.1) return data.slice();
            
            const kernelSize = Math.ceil(sigma * 3) * 2 + 1;
            const kernel = [];
            const half = Math.floor(kernelSize / 2);
            let sum = 0;
            
            for (let i = 0; i < kernelSize; i++) {
                const x = i - half;
                const g = Math.exp(-(x * x) / (2 * sigma * sigma));
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
        
        function createGlobe() {
            // Earth base sphere (dark ocean)
            const geometry = new THREE.SphereGeometry(GLOBE_RADIUS, 64, 64);
            const material = new THREE.MeshPhongMaterial({
                color: 0x0a0a1a,
                shininess: 5
            });
            return new THREE.Mesh(geometry, material);
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
            
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);
            
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.minDistance = GLOBE_RADIUS * 1.5;
            controls.maxDistance = GLOBE_RADIUS * 10;
            controls.autoRotate = false;
            controls.autoRotateSpeed = 0.5;
            
            // Lighting
            scene.add(new THREE.AmbientLight(0xffffff, 0.3));
            
            const sunLight = new THREE.DirectionalLight(0xffffff, 1.0);
            sunLight.position.set(10, 5, 10);
            scene.add(sunLight);
            
            const fillLight = new THREE.DirectionalLight(0x4466aa, 0.3);
            fillLight.position.set(-5, -5, -5);
            scene.add(fillLight);
            
            // Create globe
            globeMesh = createGlobe();
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
            for (const res of Object.keys(densityData.histograms).sort((a, b) => +a - +b)) {
                const opt = document.createElement('option');
                opt.value = res;
                opt.textContent = resLabels[res] || `${res} bins`;
                resSelect.appendChild(opt);
            }
            
            const defaultRes = Object.keys(densityData.histograms).includes('360') ? '360' : 
                              Object.keys(densityData.histograms)[0];
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
            
            document.getElementById('focus-btn').addEventListener('click', focusOnUSA);
            
            document.getElementById('point-count').textContent = 
                densityData.total_points.toLocaleString();
        }
        
        async function init() {
            initScene();
            
            try {
                const compressed = Uint8Array.from(atob(EMBEDDED_DATA), c => c.charCodeAt(0));
                const jsonText = pako.ungzip(compressed, { to: 'string' });
                densityData = JSON.parse(jsonText);
                
                console.log('Loaded globe data:', {
                    resolutions: Object.keys(densityData.histograms),
                    totalPoints: densityData.total_points,
                    boundaries: densityData.boundaries?.length || 0
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
    parser.add_argument('--sigma', type=float, default=0.1,
                        help='Default smoothing sigma (0-0.5, lower = sharper)')
    parser.add_argument('--power', type=float, default=2.0,
                        help='Default power exponent')
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
