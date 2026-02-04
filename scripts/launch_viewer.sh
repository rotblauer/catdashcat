#!/bin/bash
# Launch the interactive 3D density viewer
#
# Prerequisites:
#   python precompute_density.py -i output/raw.tsv.gz -o output/viewer/density_data.json.gz
#
# Usage:
#   ./scripts/launch_viewer.sh
#   ./scripts/launch_viewer.sh 9000  # use custom port
#
# Then open http://localhost:8888 in your browser

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VIEWER_DIR="$PROJECT_DIR/output/viewer"
PORT="${1:-8888}"

# Check if density data exists
if [ ! -f "$VIEWER_DIR/density_data.json.gz" ] && [ ! -f "$VIEWER_DIR/density_data.json" ]; then
    echo "⚠️  Density data not found. Generating..."
    cd "$PROJECT_DIR"
    .venv/bin/python precompute_density.py -i output/raw.tsv.gz -o output/viewer/density_data.json.gz --resolutions 100 250 500
fi

# Kill any existing server on this port
lsof -ti:$PORT | xargs kill -9 2>/dev/null || true

echo "🚀 Starting local server at http://localhost:$PORT"
echo "   Press Ctrl+C to stop"
echo ""

cd "$VIEWER_DIR"
python3 -m http.server $PORT
