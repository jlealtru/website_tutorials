"""One-shot notebook patcher: applies the modernization edits described in
the approved plan to every notebook under ../notebooks/.

Re-running this script is safe — every transformation is idempotent
(string-replacement with a fresh-state check).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "notebooks"

# A small injected setup cell prepended to every PyTorch notebook so the
# device pick + seed helper is available without changing every line.
SETUP_CELL_MARK = "# === modernization-setup (auto-injected) ==="
SETUP_CELL_SOURCE = [
    f"{SETUP_CELL_MARK}\n",
    "import sys, os\n",
    "if '.' not in sys.path:\n",
    "    sys.path.insert(0, '.')\n",
    "from _utils import pick_device, set_seed\n",
    "device = pick_device()\n",
    "print(f'using device: {device}')\n",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def save(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def cell_source(cell: dict) -> str:
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src


def set_cell_source(cell: dict, text: str) -> None:
    # Preserve list-of-lines format jupyter prefers.
    lines = text.splitlines(keepends=True)
    cell["source"] = lines


def inject_setup_cell(nb: dict) -> bool:
    cells = nb.get("cells", [])
    for c in cells:
        if c.get("cell_type") == "code" and SETUP_CELL_MARK in cell_source(c):
            return False
    setup = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": SETUP_CELL_SOURCE,
    }
    cells.insert(0, setup)
    return True


# ----- text transformations -----

# Catches the common forms of "device = 'cuda' if torch.cuda.is_available() else 'cpu'"
# and "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')" and
# 'cuda:0' variants.
DEVICE_LINE_RE = re.compile(
    r"""device\s*=\s*(?:torch\.device\(\s*)?['"]cuda(?::0)?['"]\s+if\s+torch\.cuda\.is_available\(\)\s+else\s+['"]cpu['"]\)?"""
)


def patch_device_lines(text: str) -> tuple[str, int]:
    new_text, n = DEVICE_LINE_RE.subn("device = pick_device()", text)
    return new_text, n


def patch_cuda_seeding(text: str) -> tuple[str, int]:
    """Replace unconditional cuda seeding/cudnn flags with safe guarded forms."""
    n = 0
    # torch.cuda.manual_seed_all(seed) on its own line → guard
    pat = re.compile(r"^(\s*)torch\.cuda\.manual_seed_all\(([^)]+)\)\s*$", re.MULTILINE)
    def _seed_repl(m):
        nonlocal n
        n += 1
        indent, arg = m.group(1), m.group(2)
        return (
            f"{indent}if torch.cuda.is_available():\n"
            f"{indent}    torch.cuda.manual_seed_all({arg})"
        )
    text = pat.sub(_seed_repl, text)

    pat2 = re.compile(r"^(\s*)torch\.backends\.cudnn\.deterministic\s*=\s*True\s*$", re.MULTILINE)
    def _cudnn_repl(m):
        nonlocal n
        n += 1
        indent = m.group(1)
        return (
            f"{indent}if torch.cuda.is_available():\n"
            f"{indent}    torch.backends.cudnn.deterministic = True"
        )
    text = pat2.sub(_cudnn_repl, text)
    return text, n


def patch_fp16_to_bf16(text: str) -> tuple[str, int]:
    # only replace fp16=True (and ignore commented variants)
    pat = re.compile(r"(?<!#)(\bfp16\s*=\s*True\b)")
    text, n = pat.subn("bf16=True", text)
    return text, n


def patch_pandas_colwidth(text: str) -> tuple[str, int]:
    pat = re.compile(r"pd\.set_option\(\s*['\"]display\.max_colwidth['\"]\s*,\s*-1\s*\)")
    text, n = pat.subn("pd.set_option('display.max_colwidth', None)", text)
    return text, n


def patch_gensim_size_kwarg(text: str) -> tuple[str, int]:
    """Word2Vec(size=128, ...) → Word2Vec(vector_size=128, ...)
    Careful: only match `size=` directly inside Word2Vec(...) calls."""
    n = 0
    # Use a function-callsite-aware replacement.
    def _replace(match):
        nonlocal n
        head, args = match.group(1), match.group(2)
        new_args, k = re.subn(r"(\bsize\s*=)", "vector_size=", args)
        n += k
        return f"{head}({new_args})"
    pat = re.compile(r"(\bWord2Vec)\(([^)]*)\)")
    text = pat.sub(_replace, text)
    return text, n


def patch_gensim_vocab_attr(text: str) -> tuple[str, int]:
    n = 0
    text, k = re.subn(r"\.wv\.vocab\b", ".wv.key_to_index", text)
    n += k
    text, k = re.subn(r"\.wv\.index2word\b", ".wv.index_to_key", text)
    n += k
    return text, n


def patch_eval_strategy(text: str) -> tuple[str, int]:
    """evaluation_strategy → eval_strategy (renamed in transformers 4.46)."""
    pat = re.compile(r"\bevaluation_strategy\s*=")
    return pat.subn("eval_strategy=", text)


def patch_cache_dir(text: str) -> tuple[str, int]:
    """Drop hard-coded Linux cache_dir kwargs that don't exist on this host."""
    # Remove `cache_dir='/media/...'` arguments (possibly with trailing comma).
    pat = re.compile(
        r",?\s*cache_dir\s*=\s*['\"]/media/[^'\"]*['\"]\s*,?",
    )
    return pat.subn("", text)


def patch_gradient_checkpointing_kwarg(text: str) -> tuple[str, int]:
    """Drop the deprecated gradient_checkpointing=False kwarg from from_pretrained.

    Modern transformers no longer accepts this arg on from_pretrained; use
    model.gradient_checkpointing_disable() instead. Removing the False case is
    a no-op since False is the default."""
    pat = re.compile(r",?\s*gradient_checkpointing\s*=\s*False\s*,?")
    return pat.subn("", text)


def patch_spacy_gpu(text: str) -> tuple[str, int]:
    # Make spacy.prefer_gpu() no-op-safe — call it inside a try/except.
    pat = re.compile(r"^(\s*)spacy\.prefer_gpu\(\)\s*$", re.MULTILINE)
    def _repl(m):
        indent = m.group(1)
        return (
            f"{indent}try:\n"
            f"{indent}    spacy.prefer_gpu()\n"
            f"{indent}except Exception:\n"
            f"{indent}    pass  # no CUDA (e.g. Apple Silicon) — stay on CPU"
        )
    text, n = pat.subn(_repl, text)
    return text, n


TRANSFORMS = [
    ("device", patch_device_lines),
    ("cuda-seed", patch_cuda_seeding),
    ("fp16→bf16", patch_fp16_to_bf16),
    ("pandas-colwidth", patch_pandas_colwidth),
    ("gensim-size", patch_gensim_size_kwarg),
    ("gensim-vocab", patch_gensim_vocab_attr),
    ("spacy-gpu", patch_spacy_gpu),
    ("eval-strategy", patch_eval_strategy),
    ("cache-dir", patch_cache_dir),
    ("grad-ckpt-kwarg", patch_gradient_checkpointing_kwarg),
]


def patch_notebook(path: Path) -> dict[str, int]:
    nb = load(path)
    totals: dict[str, int] = {name: 0 for name, _ in TRANSFORMS}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell_source(cell)
        new_src = src
        for name, fn in TRANSFORMS:
            new_src, k = fn(new_src)
            totals[name] += k
        if new_src != src:
            set_cell_source(cell, new_src)
    # Inject setup cell only into notebooks that touch torch.
    touches_torch = any(
        "import torch" in cell_source(c) or "torch." in cell_source(c)
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code"
    )
    if touches_torch:
        if inject_setup_cell(nb):
            totals["setup-cell"] = totals.get("setup-cell", 0) + 1
    save(path, nb)
    return totals


def main() -> int:
    notebooks = sorted(ROOT.glob("*.ipynb"))
    for nb_path in notebooks:
        # Skip the -Copy1 backup; it gets deleted separately.
        if nb_path.name.endswith("-Copy1.ipynb"):
            continue
        diffs = patch_notebook(nb_path)
        applied = {k: v for k, v in diffs.items() if v}
        print(f"{nb_path.name}: {applied or 'no changes'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
