#!/usr/bin/env python3
"""Compile report/relatorio_a2c.typ to PDF and report the page count.

The PyPI ``typst`` package ships the compiler as a Python binding without a CLI
entry point, so this is the ``typst compile`` of this repo.

    python tools/build_report_a2c.py
"""
from pathlib import Path

import typst

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "report" / "relatorio_a2c.typ"
OUT = REPO_ROOT / "report" / "relatorio_a2c.pdf"


def main() -> int:
    typst.compile(str(SRC), output=str(OUT), root=str(REPO_ROOT))
    pages = OUT.read_bytes().count(b"/Type /Page\n") or OUT.read_bytes().count(b"/Type/Page")
    print(f"{OUT.relative_to(REPO_ROOT)}  {OUT.stat().st_size / 1024:.0f} KB  ~{pages} páginas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
