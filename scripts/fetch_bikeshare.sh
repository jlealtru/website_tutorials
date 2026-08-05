#!/usr/bin/env bash
set -euo pipefail
DEST="data/capital_bikes"
BASE="https://s3.amazonaws.com/capitalbikeshare-data"
mkdir -p "$DEST"
for y in 2019 2020; do
  for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    f="${y}${m}-capitalbikeshare-tripdata.zip"
    if [ -f "$DEST/$f" ]; then echo "skip $f"; continue; fi
    echo "get  $f"
    curl -sS -f -o "$DEST/$f" "$BASE/$f" || { echo "FAIL $f"; exit 1; }
  done
done
echo "DONE $(ls "$DEST"/*.zip | wc -l | tr -d ' ') zips, $(du -sh "$DEST" | cut -f1)"
