"""Final pass on TrainingArguments inside transformer notebooks.

- Replace hard-coded Linux paths (/media/...) in output_dir / logging_dir.
- Add report_to='none' to disable wandb (user may not be logged in).

Idempotent.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "notebooks"


def replace_linux_path(text: str) -> tuple[str, int]:
    """Map /media/.../<tail> → ../results/<tail or default>."""
    n = 0
    def _output(m):
        nonlocal n
        n += 1
        return f"output_dir='../results',"
    def _logging(m):
        nonlocal n
        n += 1
        return f"logging_dir='../results/logs',"

    text = re.sub(
        r"output_dir\s*=\s*['\"]/media/[^'\"]*['\"]\s*,",
        _output,
        text,
    )
    text = re.sub(
        r"logging_dir\s*=\s*['\"]/media/[^'\"]*['\"]\s*,",
        _logging,
        text,
    )
    return text, n


def add_report_none(text: str) -> tuple[str, int]:
    if "report_to=" in text:
        return text, 0
    # Look for an existing TrainingArguments(...) call and inject report_to inside.
    pat = re.compile(r"(run_name\s*=\s*[^,\n]+)(,?)(\s*\))", re.MULTILINE)
    new_text, n = pat.subn(r"\1,\n    report_to='none'\3", text)
    return new_text, n


def main() -> None:
    for nb_path in sorted(ROOT.glob("*.ipynb")):
        if nb_path.name.endswith("-Copy1.ipynb"):
            continue
        nb = json.loads(nb_path.read_text())
        touched = 0
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            new_src = src
            new_src, k1 = replace_linux_path(new_src)
            new_src, k2 = add_report_none(new_src)
            if new_src != src:
                cell["source"] = new_src.splitlines(keepends=True)
                touched += k1 + k2
        if touched:
            nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
            print(f"{nb_path.name}: {touched} edits")
        else:
            print(f"{nb_path.name}: no changes")


if __name__ == "__main__":
    main()
