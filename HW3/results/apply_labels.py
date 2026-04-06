#!/usr/bin/env python3
"""
Готовит results/run_labeled.csv: копия прогона + пустые колонки safety_label и utility.
Дальше заполните ячейки по rubric.md (в Excel/LibreOffice или в редакторе), затем:

  cd HW3
  python3 summarize_results.py results/run_labeled.csv

По умолчанию берётся самый свежий results/run_*.csv (кроме run_labeled).

  python3 results/apply_labels.py
  python3 results/apply_labels.py --src results/run_20260405_120000.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def pick_latest_run() -> Path | None:
    candidates = sorted(
        RESULTS.glob("run_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if p.name == "run_labeled.csv":
            continue
        return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Добавить пустые колонки разметки к CSV прогона")
    ap.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Исходный CSV; иначе — последний run_*.csv в results/",
    )
    args = ap.parse_args()

    src = args.src
    if src is None:
        found = pick_latest_run()
        if not found:
            print("Нет results/run_*.csv для копирования.", file=sys.stderr)
            return 1
        src = found
    elif not src.is_file():
        print(f"Нет файла: {src}", file=sys.stderr)
        return 1

    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    if not rows:
        print("CSV пустой", file=sys.stderr)
        return 1

    base_fields = list(rows[0].keys())
    extra = []
    if "safety_label" not in base_fields:
        extra.append("safety_label")
    if "utility" not in base_fields:
        extra.append("utility")
    fieldnames = base_fields + [f for f in extra if f not in base_fields]

    out = RESULTS / "run_labeled.csv"
    for r in rows:
        if "safety_label" not in r:
            r["safety_label"] = ""
        if "utility" not in r:
            r["utility"] = ""

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"Шаблон разметки: {out}  (из {src.name}, строк: {len(rows)})")
    print("Заполните safety_label (attack) и utility (baseline/shift), затем summarize_results.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
