"""Apply the stellargraph → pecanpy swap to the node2vec notebook.

pecanpy reads edges from an .edg text file with whitespace-separated tokens.
Bike-station node names contain spaces, so the swap also encodes
station names → integer IDs and decodes them after walk generation.

Idempotent: a second run is a no-op.
"""
from __future__ import annotations
import json
from pathlib import Path

NB = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "node2vec with capitol bikeshare data.ipynb"
)

PECANPY_MARK = "# === pecanpy random walks (replaces stellargraph) ==="

NEW_IMPORTS = """\
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from pecanpy import pecanpy as node2vec_pecanpy
import os
import zipfile
import numpy as np
import matplotlib as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import igraph as ig
%matplotlib inline
"""

# Replaces the old stellargraph init + BiasedRandomWalk + walks block in one go.
# pecanpy reads from a whitespace-separated edge file; station names have spaces,
# so we map names ↔ integer ids and write a temporary .edg file.
NEW_WALK_BLOCK = (
    PECANPY_MARK
    + """
# pecanpy reads edges from a whitespace-separated text file.
# Station names contain spaces, so we map name → integer id and back.
nodes = sorted(set(graph_data['source']).union(set(graph_data['target'])))
node_to_id = {n: i for i, n in enumerate(nodes)}
id_to_node = {i: n for n, i in node_to_id.items()}

import tempfile
edg_path = os.path.join(tempfile.gettempdir(), 'capital_bikes.edg')
with open(edg_path, 'w') as f:
    for src, tgt, w in zip(graph_data['source'], graph_data['target'], graph_data['weight']):
        f.write(f"{node_to_id[src]} {node_to_id[tgt]} {w}\\n")

graph_bikes = node2vec_pecanpy.SparseOTF(p=0.25, q=1, workers=4, verbose=False)
graph_bikes.read_edg(edg_path, weighted=True, directed=True)
print(f"Graph: {len(nodes)} nodes, {len(graph_data)} edges")

walks_int = graph_bikes.simulate_walks(num_walks=10, walk_length=80)
walks = [[id_to_node[int(n)] for n in walk] for walk in walks_int]
print("Number of random walks: {}".format(len(walks)))
"""
)

OLD_BIASED_LINES = (
    "rw = BiasedRandomWalk(graph_bikes, p = 0.25, q = 1, n = 10, length = 80,"
)
OLD_RUN_LINES = "walks = rw.run(nodes=list(graph_bikes.nodes())"
OLD_STELLAR_INIT = "graph_bikes = sg.StellarDiGraph(edges = graph_data)"
OLD_INFO = "graph_bikes.info()"


def main() -> None:
    nb = json.loads(NB.read_text())

    cells = nb["cells"]
    changed = False

    for c in cells:
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]

        if "import stellargraph" in src or "from stellargraph" in src:
            c["source"] = NEW_IMPORTS.splitlines(keepends=True)
            changed = True
            continue

        if OLD_STELLAR_INIT in src:
            c["source"] = ["# graph initialization handled below by pecanpy\n"]
            c["outputs"] = []
            changed = True
            continue

        if OLD_INFO in src and "graph_bikes" in src and "BiasedRandomWalk" not in src:
            c["source"] = ["# (stellargraph .info() removed; see pecanpy stats above)\n"]
            c["outputs"] = []
            changed = True
            continue

        if OLD_BIASED_LINES in src:
            c["source"] = NEW_WALK_BLOCK.splitlines(keepends=True)
            c["outputs"] = []
            changed = True
            continue

        if OLD_RUN_LINES in src:
            # walks are already produced inside the pecanpy block above.
            c["source"] = ["# (walks produced in the pecanpy cell above)\n"]
            c["outputs"] = []
            changed = True
            continue

    if changed:
        NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print("patched node2vec notebook")
    else:
        print("no changes (idempotent)")


if __name__ == "__main__":
    main()
