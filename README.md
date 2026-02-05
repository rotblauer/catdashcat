# CatDash - Interactive 3D Globe Viewer for GPS Data

> ⚠️ **VIBE CODED** ⚠️
> 
> Ship it. 🚀

Generate stunning, self-contained 3D globe visualizations from GPS tracking data. The output is a single HTML file that runs entirely in the browser - no server required. Perfect for sharing, GitHub Pages, or serving from simple hardware like a Raspberry Pi.

![Globe Viewer Screenshot](ex.globe.png)

## Features

- 🌍 **Interactive 3D Globe** - Pan, zoom, rotate with mouse/touch
- 📊 **Multi-resolution density peaks** - See patterns at different scales  
- 🗺️ **Natural Earth base maps** - Beautiful satellite-style backgrounds
- 🏔️ **Top 10 Peak Locations** - Click to zoom into 100km local views
- 🗾 **OpenStreetMap overlays** - Geographic context in local views
- 📊 **Satellite Bar Chart** - Floating 3D chart showing US state counts
- 📁 **Self-contained HTML** - Single ~5MB file, works offline
- ⚡ **WebGL accelerated** - Smooth 60fps with millions of points
- 🔧 **Real-time controls** - Adjust smoothing, peak height, threshold

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate input data (from GPS JSON stream)
./run.sh < input.json

# 3. Generate interactive globe viewer
.venv/bin/python generate_globe_viewer.py \
    -i output/raw.tsv.gz \
    --resolutions 720 1440 \
    --n-peaks 10 \
    --workers 4

# 4. Open in browser (default output: docs/index.html)
open docs/index.html
```

## Command Line Options

```bash
.venv/bin/python generate_globe_viewer.py [OPTIONS]

Options:
  -i, --input           Input TSV file (default: output/raw.tsv.gz)
  -o, --output          Output HTML file (default: docs/index.html)
  --resolutions         Resolution levels to compute (default: 180 360 720 1440 2880)
                        Higher = more detail. 360 = 1° bins, 1440 = 0.25° bins
  --sigma               Default smoothing (0-0.5, default: 0)
                        Lower = sharper peaks, 0 = raw data
  --power               Peak height exponent (default: 2.0)
                        Higher = more dramatic peaks
  --n-peaks             Number of top peaks for local views (default: 10)
  --local-radius        Radius in km for local views (default: 50 = 100km×100km)
  --local-resolution    Resolution for local histograms (default: 200)
  --workers             Parallel workers for local histograms (default: 4)
  --chunk-size          Rows per chunk when reading (default: 500,000)
  --shapefile           US state/county boundaries shapefile
```

## Examples

```bash
# High resolution with sharp peaks
.venv/bin/python generate_globe_viewer.py \
    --resolutions 720 1440 2880 \
    --sigma 0.01 \
    --n-peaks 10 \
    --workers 5

# Quick preview (lower resolution)
.venv/bin/python generate_globe_viewer.py \
    --resolutions 180 360 \
    --n-peaks 5

# Smooth terrain visualization
.venv/bin/python generate_globe_viewer.py \
    --resolutions 360 720 \
    --sigma 0.3 \
    --power 1.5
```

## Viewer Controls

### Globe View
- **Resolution** - Switch between computed detail levels
- **Smoothing (σ)** - 0 = raw, higher = smoother terrain
- **Peak Height** - Exponent for peak emphasis
- **Extrusion Scale** - Vertical height multiplier
- **Threshold** - Minimum density to display
- **Base Map** - Dark / Natural Earth / Topography
- **Auto Rotate** - Continuous slow rotation
- **Toggle State Chart** - Show/hide floating US state counts bar chart

### Local Peak Views
Click any peak in the sidebar to zoom into a 100km×100km region:
- **3D density visualization** of the local area
- **OpenStreetMap overlay** option for geographic context
- **Independent controls** for local smoothing/height

### Mouse Controls
- **Drag** - Rotate globe
- **Scroll** - Zoom in/out
- **Right-drag** - Pan view

## Data Pipeline

### 1. Generate Raw Data

The input pipeline uses Go for high-performance JSON parsing:

```bash
# Compile Go parser (one-time)
go build -o catdash main.go

# Stream GPS JSON to TSV
./catdash < gps_data.json > output/raw.tsv

# Or use the convenience script
./run.sh < gps_data.json
```

Input JSON format (newline-delimited GeoJSON features):
```json
{"type":"Feature","geometry":{"type":"Point","coordinates":[-93.25,44.98]},"properties":{"Time":"2024-01-15T14:30:00Z","Speed":2.5,"Name":"Luna"}}
```

### 2. Generate Viewer

```bash
.venv/bin/python generate_globe_viewer.py -i output/raw.tsv.gz
```

Output: `docs/index.html` (self-contained, ready for GitHub Pages)

### 3. Deploy

The output HTML file is completely self-contained:
- Copy to any web server
- Open directly in browser (file://)
- Host on GitHub Pages
- Serve from Raspberry Pi

## Input Data Format

Tab-separated file with columns:
- `lat` - Latitude (required)
- `lon` - Longitude (required)
- Additional columns are ignored

## Boundary Data

For US state/county boundaries, download from Census Bureau:
```bash
curl -L -o output/cb_2020_us_county_500k.zip \
    "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"
unzip output/cb_2020_us_county_500k.zip -d output/
```

For world country boundaries (Natural Earth):
```bash
curl -L -o output/ne_110m_admin_0_countries.zip \
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
unzip output/ne_110m_admin_0_countries.zip -d output/
```

## Requirements

- Python 3.10+
- pandas, numpy, geopandas
- Modern browser with WebGL support

## Performance

Optimized for millions of data points:
- Chunked file reading (configurable chunk size)
- Multi-threaded local histogram computation
- Sparse histogram storage (only non-zero cells)
- WebGL rendering with efficient buffer geometry

Typical performance:
- 87M points → ~30 seconds for global histograms
- 10 local views (parallel) → ~2 minutes
- Output file size: 5-6 MB

## License

MIT
