#!/usr/bin/env bash
# Fetch the Jigsaw Toxic Comment Classification Challenge data into data/jigsaw/.
# Requires a Kaggle API token at ~/.kaggle/kaggle.json (chmod 600) AND that you have
# accepted the competition rules at:
#   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules
set -euo pipefail
DEST="data/jigsaw"
mkdir -p "$DEST"
COMP="jigsaw-toxic-comment-classification-challenge"
uv run kaggle competitions download -c "$COMP" -p "$DEST"
cd "$DEST"
unzip -o "${COMP}.zip"          # -> train.csv.zip test.csv.zip test_labels.csv.zip sample_submission.csv.zip
for z in train.csv.zip test.csv.zip test_labels.csv.zip sample_submission.csv.zip; do
  [ -f "$z" ] && unzip -o "$z"
done
echo "jigsaw files:"; ls -1 *.csv
