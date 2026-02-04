#!/usr/bin/env python3
"""
Efficiently convert newline-delimited JSON features to raw.tsv.gz

Usage:
    cat input.json | python json_to_tsv.py -o output/raw.tsv.gz
    python json_to_tsv.py -i input.json -o output/raw.tsv.gz

Handles inconsistent JSON schemas with memory-efficient two-pass processing.
"""

import argparse
import json
import sys
import gzip
import csv
import tempfile
import os
from collections import OrderedDict


def extract_feature(line):
    """
    Extract properties and lat/lon from a GeoJSON feature line.
    Returns (props_dict, lat, lon) or None if invalid.
    """
    line = line.strip()
    if not line:
        return None
    try:
        feature = json.loads(line)
        props = feature.get('properties', {})
        geom = feature.get('geometry', {})
        coords = geom.get('coordinates', [])

        # Extract lat/lon
        if len(coords) >= 2:
            try:
                lon = float(coords[0])
                lat = float(coords[1])
            except (TypeError, ValueError):
                return None
        else:
            return None

        return props, lat, lon
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def process_two_pass(input_path, output_path):
    """
    Memory-efficient two-pass processing for file input.

    Pass 1: Scan all records to find all columns (only stores column names)
    Pass 2: Write all records with consistent schema
    """
    all_columns = OrderedDict()
    all_columns['lat'] = True
    all_columns['lon'] = True

    # Pass 1: Scan for all columns
    print("Pass 1: Scanning for columns...", file=sys.stderr)
    total_count = 0
    skipped = 0

    with open(input_path, 'r') as f:
        for line in f:
            result = extract_feature(line)
            if result is None:
                skipped += 1
                continue

            props, lat, lon = result
            total_count += 1

            # Track new columns
            for key in props.keys():
                if key not in all_columns:
                    all_columns[key] = True

            if total_count % 100000 == 0:
                print(f"  Scanned {total_count:,} records, {len(all_columns)} columns...",
                      file=sys.stderr, end='\r')

    print(f"\n  Scanned {total_count:,} records, skipped {skipped:,}", file=sys.stderr)
    print(f"  Found {len(all_columns)} columns", file=sys.stderr)

    # Pass 2: Write with consistent schema
    print("Pass 2: Writing output...", file=sys.stderr)
    columns = list(all_columns.keys())

    written = 0
    with open(input_path, 'r') as fin:
        with gzip.open(output_path, 'wt', encoding='utf-8', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=columns, delimiter='\t',
                                    extrasaction='ignore', restval='')
            writer.writeheader()

            for line in fin:
                result = extract_feature(line)
                if result is None:
                    continue

                props, lat, lon = result
                row = props.copy()
                row['lat'] = lat
                row['lon'] = lon
                writer.writerow(row)
                written += 1

                if written % 100000 == 0:
                    print(f"  Written {written:,} records...", file=sys.stderr, end='\r')

    print(f"\n✓ Wrote {written:,} records to {output_path}", file=sys.stderr)
    return written


def process_stdin(output_path):
    """
    Process stdin by writing to temp file first, then using two-pass.
    """
    print("Reading from stdin to temp file...", file=sys.stderr)

    # Write stdin to temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        tmp_path = tmp.name
        count = 0
        for line in sys.stdin:
            tmp.write(line)
            count += 1
            if count % 100000 == 0:
                print(f"  Buffered {count:,} lines...", file=sys.stderr, end='\r')
        print(f"\n  Buffered {count:,} lines to {tmp_path}", file=sys.stderr)

    try:
        # Now use two-pass on temp file
        process_two_pass(tmp_path, output_path)
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert newline-delimited GeoJSON features to gzipped TSV')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Input JSON file (default: stdin)')
    parser.add_argument('-o', '--output', type=str, default='output/raw.tsv.gz',
                        help='Output TSV.gz file (default: output/raw.tsv.gz)')

    args = parser.parse_args()

    if args.input:
        process_two_pass(args.input, args.output)
    else:
        process_stdin(args.output)


if __name__ == '__main__':
    main()
