#!/usr/bin/env python3
"""
CatDash Geospatial Analysis Runner
==================================
Unified CLI for running all geospatial analyses on cat tracking data.

Usage:
    python run_analysis.py --input output/raw.tsv.gz --output output/results
    python run_analysis.py --input output/raw.tsv.gz --analysis all --sample 0.1
    python run_analysis.py --input output/raw.tsv.gz --analysis standard --visualize
    python run_analysis.py --input output/raw.tsv.gz --analysis creative --sample 0.1
"""

import argparse
import sys
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(
        description='CatDash Geospatial Analysis Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all analyses on full data
  python run_analysis.py -i output/raw.tsv.gz -o output/results
  
  # Run standard analysis with 10% sample
  python run_analysis.py -i output/raw.tsv.gz -o output/results -s 0.1 -a standard
  
  # Run advanced analysis with visualization
  python run_analysis.py -i output/raw.tsv.gz -o output/results -a advanced --visualize
  
  # Run creative analysis (fractal dimension, entropy, etc.)
  python run_analysis.py -i output/raw.tsv.gz -o output/results -a creative -s 0.1
  
  # Quick test with 10000 rows
  python run_analysis.py -i output/raw.tsv.gz -o output/test -n 10000
        """
    )

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input file (gzipped TSV)')
    parser.add_argument('-o', '--output', type=str, default='output/analysis_results',
                        help='Output directory')
    parser.add_argument('-a', '--analysis', type=str, default='all',
                        choices=['all', 'standard', 'advanced', 'creative'],
                        help='Which analysis suite to run')
    parser.add_argument('-s', '--sample', type=float, default=1.0,
                        help='Sample fraction (0-1) for efficiency')
    parser.add_argument('-n', '--nrows', type=int, default=None,
                        help='Number of rows to read (for testing)')
    parser.add_argument('--name', type=str, default=None,
                        help='Filter to specific Name')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations after analysis')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} not found")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CatDash Geospatial Analysis Suite")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode:   {args.analysis}")
    if args.sample < 1.0:
        print(f"Sample: {args.sample * 100:.1f}%")
    if args.nrows:
        print(f"Rows:   {args.nrows:,}")
    print("=" * 60)

    start_time = time.time()

    # Import analysis modules
    from geospatial_analysis import load_data, run_analysis
    from advanced_analysis import run_advanced_analysis
    from creative_analysis import run_creative_analysis

    # Load data
    print("\n📥 Loading data...")
    df = load_data(args.input, sample_frac=args.sample, nrows=args.nrows)

    if args.name:
        print(f"Filtering to Name={args.name}...")
        df = df[df['Name'] == args.name].copy()
        print(f"Filtered to {len(df):,} records")

    # Run analyses
    if args.analysis in ['all', 'standard']:
        print("\n" + "=" * 40)
        print("📊 Running Standard Analysis")
        print("=" * 40)
        results, df = run_analysis(df, str(output_path / 'standard'), name_filter=None)

    if args.analysis in ['all', 'advanced']:
        print("\n" + "=" * 40)
        print("🔬 Running Advanced Analysis")
        print("=" * 40)
        advanced_results = run_advanced_analysis(df, str(output_path / 'advanced'))

    if args.analysis in ['all', 'creative']:
        print("\n" + "=" * 40)
        print("🎨 Running Creative Analysis")
        print("=" * 40)
        creative_results = run_creative_analysis(df, str(output_path / 'creative'))

    # Generate visualizations
    if args.visualize:
        print("\n" + "=" * 40)
        print("📈 Generating Visualizations")
        print("=" * 40)
        from visualization_suite import generate_all_visualizations
        generate_all_visualizations(str(output_path), args.input)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"✅ Analysis complete in {elapsed:.1f} seconds")
    print(f"📁 Results saved to: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
