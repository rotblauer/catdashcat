#!/usr/bin/env python3
"""
Efficiently convert newline-delimited JSON features to raw.tsv.gz

Usage:
    cat input.json | python json_to_tsv.py -o output/raw.tsv.gz
    python json_to_tsv.py -i input.json -o output/raw.tsv.gz

Uses geopandas to robustly handle inconsistent JSON schemas.
"""

import argparse
import json
import sys
import geopandas


def parse_features(input_stream):
    """
    Parse newline-delimited GeoJSON features into a DataFrame.
    Uses geopandas which handles inconsistent schemas robustly.
    """
    print("Reading JSON features...", file=sys.stderr)
    features = []
    count = 0
    for line in input_stream:
        line = line.strip()
        if not line:
            continue
        try:
            features.append(json.loads(line))
            count += 1
            if count % 100000 == 0:
                print(f"  Read {count:,} features...", file=sys.stderr, end='\r')
        except json.JSONDecodeError:
            continue

    print(f"\n  Read {count:,} total features", file=sys.stderr)

    print("Converting to GeoDataFrame...", file=sys.stderr)
    df = geopandas.GeoDataFrame.from_features(features)

    # Add lat/lon columns from geometry
    df['lat'] = df.geometry.y
    df['lon'] = df.geometry.x

    # Remove geometry column
    df = df.drop(columns=['geometry'])

    print(f"  Created DataFrame with {len(df):,} rows, {len(df.columns)} columns", file=sys.stderr)
    print(f"  Columns: {list(df.columns)}", file=sys.stderr)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Convert newline-delimited GeoJSON features to gzipped TSV')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Input JSON file (default: stdin)')
    parser.add_argument('-o', '--output', type=str, default='output/raw.tsv.gz',
                        help='Output TSV.gz file (default: output/raw.tsv.gz)')

    args = parser.parse_args()

    # Read input
    if args.input:
        with open(args.input, 'r') as f:
            df = parse_features(f)
    else:
        df = parse_features(sys.stdin)

    # Write output
    print(f"Writing to {args.output}...", file=sys.stderr)
    df.to_csv(args.output, sep='\t', compression='gzip', index=False)
    print(f"✓ Wrote {len(df):,} records to {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
