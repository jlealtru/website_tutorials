#!/usr/bin/env python
"""Fetch the Pitchfork reviews dataset and materialize the files the ETM
notebooks expect under ``data/pitchfork/``.

Source: https://huggingface.co/datasets/mattismegevand/pitchfork (``reviews.csv``,
~26k reviews). That file's schema differs from what the notebooks were originally
written against, so we remap it:

    rating            -> score            (0-10 Pitchfork score)
    <none>            -> link             (synthetic, unique per row: the doc id /
                                           merge key the notebooks use)
    review/artist/album/genre             kept as-is

We also write ``stop.txt`` (one stopword per line) from spaCy's built-in English
stopword list, because the notebooks read a stopword file that never shipped with
the repo.

Run:  uv run python scripts/fetch_pitchfork.py
Everything lands in the git-ignored ``data/`` tree, so nothing here is committed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from spacy.lang.en.stop_words import STOP_WORDS

REPO_ID = "mattismegevand/pitchfork"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "pitchfork"
# columns the two ETM notebooks read from pitchfork.csv
KEEP = ["review", "link", "artist", "album", "score", "genre"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"downloading {REPO_ID}:reviews.csv …")
    src = hf_hub_download(REPO_ID, "reviews.csv", repo_type="dataset")
    df = pd.read_csv(src, low_memory=False)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}")

    # --- remap to the notebooks' expected schema ---
    df = df.rename(columns={"rating": "score"})
    # `link` is used purely as a unique document id / left-join key; the HF export
    # has no review URL, so synthesize a stable unique id per row.
    df.insert(0, "link", [f"pf_{i:06d}" for i in range(len(df))])

    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        raise SystemExit(f"source is missing expected columns: {missing}")

    # keep the notebook columns first, then any extras (harmless to carry along)
    ordered = KEEP + [c for c in df.columns if c not in KEEP]
    df = df[ordered]

    csv_path = OUT_DIR / "pitchfork.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"wrote {csv_path}  ({len(df):,} rows)")

    # --- stopwords file the notebooks read (never shipped in the repo) ---
    stop_path = OUT_DIR / "stop.txt"
    stop_path.write_text("\n".join(sorted(STOP_WORDS)), encoding="utf-8")
    print(f"wrote {stop_path}  ({len(STOP_WORDS)} spaCy English stopwords)")


if __name__ == "__main__":
    main()
