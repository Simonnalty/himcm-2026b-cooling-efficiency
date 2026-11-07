# -*- coding: utf-8 -*-
"""Build paper_build.tex (inject table fragments, avoiding \\input inside
tabular which breaks under the 2020+ LaTeX kernel), then compile twice."""
import os, re, subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, ".."))
PAPER = os.path.join(ROOT, "paper")

src = open(os.path.join(PAPER, "paper.tex"), encoding="utf-8").read()
for ph, name in [("%%TABLE_BUS%%", "t_bus"), ("%%TABLE_STATION%%", "t_station"),
                 ("%%TABLE_EFF%%", "t_eff"), ("%%TABLE_VENUE%%", "t_venue"),
                 ("%%TABLE_TORNADO%%", "t_tornado")]:
    frag = open(os.path.join(PAPER, "gen", name + ".tex"), encoding="utf-8").read()
    src = src.replace(ph, frag.replace("\r\n", "\n"))
out = os.path.join(PAPER, "paper_build.tex")
open(out, "w", encoding="utf-8", newline="\n").write(src)
assert "%%TABLE" not in src

for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "paper_build.tex"],
                   cwd=PAPER, capture_output=True)
log = open(os.path.join(PAPER, "paper_build.log"), encoding="utf-8", errors="ignore").read()
pages = re.findall(r"Output written on paper_build.pdf \((\d+) pages", log)
errs = [l for l in log.splitlines() if l.startswith("! ")]
print("pages:", pages if pages else "NO PDF", "| errors:", len(errs))
for e in errs[:6]:
    print(" ", e)
