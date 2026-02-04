#!/usr/bin/env python3
"""Generate a self-contained static HTML viewer with embedded density data.

This creates a single HTML file that can be opened directly in a browser
without needing a web server - perfect for sharing or hosting on GitHub Pages.

Example usage:
    python generate_static_viewer.py -i output/raw.tsv.gz -o output/viewer/density_viewer.html
    python generate_static_viewer.py --resolutions 100 250 500 --sigma 1.2 --power 2.5
    .venv/bin/python generate_static_viewer.py --resolutions 100 250 500
"""

import argparse
import base64
import gzip
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Continental US bounds
US_LAT_MIN, US_LAT_MAX = 24.5, 49.5
US_LON_MIN, US_LON_MAX = -125.0, -66.5

DEFAULT_RESOLUTIONS = [100, 250, 500]
DEFAULT_CHUNK_SIZE = 500_000


def build_histograms_multi_resolution(input_file: str, resolutions: list, chunk_size: int) -> tuple:
    """Build 2D histograms at multiple resolutions in a single pass."""
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

    state_polys = {}
    skip_states = {'02', '15', '72', '78', '60', '66', '69'}

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

                if (coords[:, 0].max() < US_LON_MIN or coords[:, 0].min() > US_LON_MAX or
                    coords[:, 1].max() < US_LAT_MIN or coords[:, 1].min() > US_LAT_MAX):
                    continue

                step = max(1, len(coords) // 50)
                simplified = coords[::step].tolist()

                if state_fip not in state_polys:
                    state_polys[state_fip] = []
                state_polys[state_fip].append(simplified)

    boundaries = []
    for state, polys in state_polys.items():
        for poly in polys:
            if len(poly) >= 3:
                boundaries.append(poly)

    return boundaries


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


def generate_html(density_data: dict, default_sigma: float = 1.2, default_power: float = 2.5) -> str:
    """Generate the complete static HTML with embedded data."""

    # Compress and encode the density data
    json_str = json.dumps(density_data, separators=(',', ':'))
    compressed = gzip.compress(json_str.encode('utf-8'))
    encoded_data = base64.b64encode(compressed).decode('ascii')

    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cat GPS Density Map - Interactive 3D Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ed 100%);
            color: #333;
            overflow: hidden;
        }
        
        #container { width: 100vw; height: 100vh; position: relative; }
        #canvas-container { width: 100%; height: 100%; }
        
        #controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.1);
            min-width: 280px;
            z-index: 100;
        }
        
        #controls h1 { font-size: 18px; margin-bottom: 5px; color: #e94560; }
        #controls .subtitle { font-size: 12px; color: #666; margin-bottom: 20px; }
        
        .control-group { margin-bottom: 18px; }
        .control-group label {
            display: block;
            font-size: 12px;
            color: #555;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .control-group input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }
        
        .control-group input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #e94560;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(233, 69, 96, 0.4);
        }
        
        .control-group .value { font-size: 14px; color: #333; font-weight: 600; margin-top: 4px; }
        
        .control-group select {
            width: 100%;
            padding: 8px 12px;
            border-radius: 6px;
            background: #f5f5f5;
            border: 1px solid #ddd;
            color: #333;
            font-size: 14px;
            cursor: pointer;
        }
        
        .control-group select:focus { outline: none; border-color: #e94560; }
        
        #stats {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.9);
            padding: 12px 18px;
            border-radius: 8px;
            font-size: 12px;
            color: #666;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        #stats span { color: #e94560; font-weight: 600; }
        
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
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        #loading p { color: #666; font-size: 14px; }
        .hidden { display: none !important; }
        
        #colorbar {
            position: absolute;
            right: 30px;
            top: 50%;
            transform: translateY(-50%);
            width: 20px;
            height: 200px;
            border-radius: 4px;
            background: linear-gradient(to top,
                #f0f4f8 0%, #b8c5d4 15%, #6b8fb8 30%,
                #e94560 50%, #ff6b6b 70%, #feca57 85%, #fff9db 100%
            );
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            border: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        #colorbar-labels {
            position: absolute;
            right: 60px;
            top: 50%;
            transform: translateY(-50%);
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            font-size: 11px;
            color: #666;
        }
        
        .hint { font-size: 10px; color: #888; margin-top: 2px; font-style: italic; }
    </style>
</head>
<body>
    <div id="container">
        <div id="canvas-container"></div>
        
        <div id="loading">
            <div class="spinner"></div>
            <p>Loading density data...</p>
        </div>
        
        <div id="controls" class="hidden">
            <h1>🐱 Cat GPS Density</h1>
            <p class="subtitle">Continental United States</p>
            
            <div class="control-group">
                <label>Resolution</label>
                <select id="resolution"></select>
            </div>
            
            <div class="control-group">
                <label>Smoothing (σ)</label>
                <input type="range" id="sigma" min="0.1" max="5" step="0.1" value="''' + str(default_sigma) + '''">
                <div class="value" id="sigma-value">''' + str(default_sigma) + '''</div>
                <div class="hint">Lower = sharper peaks, Higher = smoother terrain</div>
            </div>
            
            <div class="control-group">
                <label>Peak Emphasis (power)</label>
                <input type="range" id="power" min="1" max="5" step="0.1" value="''' + str(default_power) + '''">
                <div class="value" id="power-value">''' + str(default_power) + '''</div>
                <div class="hint">Higher = more dramatic peaks</div>
            </div>
            
            <div class="control-group">
                <label>Height Scale</label>
                <input type="range" id="height-scale" min="0.1" max="3" step="0.1" value="1.0">
                <div class="value" id="height-scale-value">1.0</div>
            </div>
            
            <div class="control-group">
                <label>Show Boundaries</label>
                <select id="show-boundaries">
                    <option value="true" selected>Yes</option>
                    <option value="false">No</option>
                </select>
            </div>
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
        let densityMesh, boundaryLines;
        let densityData = null;
        
        const settings = {
            resolution: 250,
            sigma: ''' + str(default_sigma) + ''',
            power: ''' + str(default_power) + ''',
            heightScale: 1.0,
            showBoundaries: true
        };
        
        const colorStops = [
            { pos: 0.00, color: [240, 244, 248] },
            { pos: 0.15, color: [184, 197, 212] },
            { pos: 0.30, color: [107, 143, 184] },
            { pos: 0.50, color: [233, 69, 96] },
            { pos: 0.70, color: [255, 107, 107] },
            { pos: 0.85, color: [254, 202, 87] },
            { pos: 1.00, color: [255, 249, 219] }
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
            
            const temp = new Float32Array(width * height);
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    let val = 0;
                    for (let k = 0; k < kernelSize; k++) {
                        const sx = Math.min(width - 1, Math.max(0, x + k - half));
                        val += data[y * width + sx] * kernel[k];
                    }
                    temp[y * width + x] = val;
                }
            }
            
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
        
        function createDensityMesh(processed) {
            const { data, width, height } = processed;
            const bounds = densityData.bounds;
            
            const geometry = new THREE.BufferGeometry();
            const vertices = [];
            const colors = [];
            const indices = [];
            
            const lonRange = bounds.lon_max - bounds.lon_min;
            const latRange = bounds.lat_max - bounds.lat_min;
            const scaleX = 10;
            const scaleY = 10 * (latRange / lonRange);
            const scaleZ = 3 * settings.heightScale;
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = y * width + x;
                    const z = data[idx];
                    
                    const px = (x / (width - 1) - 0.5) * scaleX;
                    const py = (y / (height - 1) - 0.5) * scaleY;
                    const pz = z * scaleZ;
                    
                    vertices.push(px, py, pz);
                    
                    const color = interpolateColor(z);
                    colors.push(color[0] / 255, color[1] / 255, color[2] / 255);
                }
            }
            
            for (let y = 0; y < height - 1; y++) {
                for (let x = 0; x < width - 1; x++) {
                    const i = y * width + x;
                    indices.push(i, i + width, i + 1);
                    indices.push(i + 1, i + width, i + width + 1);
                }
            }
            
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            geometry.setIndex(indices);
            geometry.computeVertexNormals();
            
            const material = new THREE.MeshLambertMaterial({
                vertexColors: true,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.95
            });
            
            return new THREE.Mesh(geometry, material);
        }
        
        function createBoundaryLines() {
            if (!densityData.boundaries || densityData.boundaries.length === 0) return null;
            
            const bounds = densityData.bounds;
            const lonRange = bounds.lon_max - bounds.lon_min;
            const latRange = bounds.lat_max - bounds.lat_min;
            const scaleX = 10;
            const scaleY = 10 * (latRange / lonRange);
            
            const group = new THREE.Group();
            
            const planeGeometry = new THREE.PlaneGeometry(scaleX * 1.05, scaleY * 1.05);
            const planeMaterial = new THREE.MeshBasicMaterial({
                color: 0xe8ecf0,
                side: THREE.DoubleSide,
                transparent: true,
                opacity: 0.5
            });
            const plane = new THREE.Mesh(planeGeometry, planeMaterial);
            plane.position.z = -0.02;
            group.add(plane);
            
            const material = new THREE.LineBasicMaterial({ color: 0x333333, transparent: false, linewidth: 2 });
            
            for (const boundary of densityData.boundaries) {
                if (boundary.length < 2) continue;
                
                const points = [];
                for (const [lon, lat] of boundary) {
                    const nx = (lon - bounds.lon_min) / lonRange - 0.5;
                    const ny = (lat - bounds.lat_min) / latRange - 0.5;
                    points.push(new THREE.Vector3(nx * scaleX, ny * scaleY, 0.01));
                }
                
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const line = new THREE.Line(geometry, material);
                group.add(line);
            }
            
            return group;
        }
        
        function updateVisualization() {
            const resKey = settings.resolution.toString();
            if (!densityData.histograms[resKey]) return;
            
            const processed = processHistogram(
                densityData.histograms[resKey],
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
        
        function initScene() {
            const container = document.getElementById('canvas-container');
            
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f4f8);
            
            const aspect = window.innerWidth / window.innerHeight;
            camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
            camera.position.set(0, -12, 10);
            camera.lookAt(0, 0, 0);
            
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);
            
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.minDistance = 5;
            controls.maxDistance = 50;
            controls.maxPolarAngle = Math.PI / 2;
            
            scene.add(new THREE.AmbientLight(0xffffff, 0.6));
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 5, 10);
            scene.add(directionalLight);
            
            const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
            backLight.position.set(-5, -5, 5);
            scene.add(backLight);
            
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
                '100': 'Low (100×100) - Fast',
                '250': 'Medium (250×250)',
                '500': 'High (500×500)',
                '1000': 'Ultra (1000×1000)'
            };
            
            resSelect.innerHTML = '';
            for (const res of Object.keys(densityData.histograms).sort((a, b) => +a - +b)) {
                const opt = document.createElement('option');
                opt.value = res;
                opt.textContent = resLabels[res] || `${res}×${res}`;
                resSelect.appendChild(opt);
            }
            
            // Pick default resolution
            const defaultRes = Object.keys(densityData.histograms).includes('250') ? '250' : 
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
                sigmaValue.textContent = settings.sigma.toFixed(1);
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
                heightValue.textContent = settings.heightScale.toFixed(1);
            });
            heightSlider.addEventListener('change', () => updateVisualization());
            
            document.getElementById('show-boundaries').addEventListener('change', (e) => {
                settings.showBoundaries = e.target.value === 'true';
                if (boundaryLines) boundaryLines.visible = settings.showBoundaries;
            });
            
            document.getElementById('point-count').textContent = 
                densityData.total_points.toLocaleString();
        }
        
        async function init() {
            initScene();
            
            try {
                // Decode embedded data
                const compressed = Uint8Array.from(atob(EMBEDDED_DATA), c => c.charCodeAt(0));
                const jsonText = pako.ungzip(compressed, { to: 'string' });
                densityData = JSON.parse(jsonText);
                
                console.log('Loaded density data:', {
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
    parser = argparse.ArgumentParser(description='Generate static HTML density viewer')
    parser.add_argument('-i', '--input', default='output/raw.tsv.gz',
                        help='Input TSV file')
    parser.add_argument('-o', '--output', default='output/viewer/density_viewer.html',
                        help='Output HTML file')
    parser.add_argument('--resolutions', type=int, nargs='+', default=DEFAULT_RESOLUTIONS,
                        help=f'Resolutions to compute (default: {DEFAULT_RESOLUTIONS})')
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Rows per chunk (default: {DEFAULT_CHUNK_SIZE:,})')
    parser.add_argument('--shapefile', default='output/cb_2020_us_county_500k.shp',
                        help='Shapefile for state boundaries')
    parser.add_argument('--sigma', type=float, default=1.2,
                        help='Default smoothing sigma')
    parser.add_argument('--power', type=float, default=2.5,
                        help='Default power exponent')
    args = parser.parse_args()

    density_data = {
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

    # Compute histograms
    print(f"\n📊 Computing histograms at {len(args.resolutions)} resolutions...")
    histograms, total_points = build_histograms_multi_resolution(
        args.input, args.resolutions, args.chunk_size
    )
    density_data['total_points'] = total_points

    for res in sorted(args.resolutions):
        hist = histograms[res]
        density_data['histograms'][str(res)] = compress_histogram(hist)
        print(f"   ✓ {res}x{res}: {(hist > 0).sum():,} non-zero cells")

    # Extract boundaries
    print(f"\n🗺️  Extracting state boundaries...")
    try:
        boundaries = extract_state_boundaries(args.shapefile)
        density_data['boundaries'] = boundaries
        print(f"   ✓ {len(boundaries)} boundary polygons")
    except Exception as e:
        print(f"   ⚠ Could not load boundaries: {e}")

    # Generate HTML
    print(f"\n📄 Generating static HTML...")
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
