#!/usr/bin/env python3
"""Write a copy of the paper laid out exactly like the Overleaf project.

The Overleaf project has main.tex, ref3.bib, old.tex, circuits/, figures/ and
generated/fig7_*.tex.  The finite-volume tables that paper/main.tex pulls in
with \\input are written into main.tex here, so the folder needs no files
beyond those the Overleaf project already has.

    python3 scripts/make_overleaf_folder.py ../overleaf_update
"""
import re
import shutil
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
INLINED = ("generated/fv_data_table", "generated/fv_fit_results")
COPIED = ["ref3.bib", "old.tex", "generated/fig7_data_table.tex",
          "generated/fig7_fit_results.tex"]


def main(target):
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    text = (PAPER / "main.tex").read_text()
    for name in INLINED:
        table = (PAPER / f"{name}.tex").read_text().strip()
        marker = f"\\input{{{name}}}"
        assert text.count(marker) == 1, marker
        text = text.replace(marker, table)
    live = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))
    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+)\}", live)))
    circuits = sorted(str(p.relative_to(PAPER)) for p in (PAPER / "circuits").glob("*.tex"))
    for name in COPIED + figures + circuits:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(PAPER / name, target / name)
    (target / "main.tex").write_text(text)
    for path in sorted(p for p in target.rglob("*") if p.is_file()):
        print(path.relative_to(target))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else PAPER.parents[1] / "overleaf_update")
