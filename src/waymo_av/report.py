"""Generate reports/report.md (+ PDF when a local engine exists) from saved metrics, registry and
figures. No numbers are typed by hand: every table is read from aggregate.py outputs.

python -m waymo_av.report --dev-tables reports/tables/dev --final-tables reports/tables/final [--rl-tables ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from waymo_av.budget import gpu_hours_used
from waymo_av.paths import REPO_ROOT, load_paths
from waymo_av.registry import git_commit


def _read(p: Path, default=""):
    return p.read_text() if p.exists() else default


def _runs_table() -> str:
    rows = ["| run | kind | status | step | params | best dev score | GPU s |", "|---|---|---|---|---|---|---|"]
    gpu = {}
    gl = REPO_ROOT / "registry" / "gpu_hours.jsonl"
    if gl.exists():
        for l in open(gl):
            r = json.loads(l)
            gpu[r["run"]] = gpu.get(r["run"], 0) + r["gpu_seconds"]
    for p in sorted((REPO_ROOT / "registry" / "runs").glob("*.json")):
        r = json.load(open(p))
        if r["name"].startswith("synthetic"):
            continue
        best = r.get("best", {})
        rows.append(f"| {r['name']} | {r.get('kind','')} | {r.get('status','')} | {r.get('step','')} | {r.get('n_params','')} | {best.get('score','') if isinstance(best, dict) else ''} | {gpu.get(r['name'], 0):.0f} |")
    return "\n".join(rows)


def build(dev_tables: Path | None, final_tables: Path | None, rl_tables: Path | None, figures: list[Path], anims: list[Path], out: Path):
    paths = load_paths()
    cal = REPO_ROOT / "manifests" / "calibration.json"
    dv = REPO_ROOT / "manifests" / "dataset_version.json"
    parts = [
        "# Waymo_AV — Robust Joint Motion Forecasting Under Missing and Noisy Observations",
        f"\nGenerated {time.strftime('%Y-%m-%d %H:%M')} from commit `{git_commit()}`. Simplified MotionLM-style two-agent joint predictor; local metrics (not official challenge scores).\n",
        "## 1. Setup\n",
        f"- Dataset: `{_read(dv, '{}').strip()}`" if dv.exists() else "- Dataset: NOT DOWNLOADED (blocked on licence login)",
        f"- Calibration: `{_read(cal, '{}').strip()[:1500]}`" if cal.exists() else "- Calibration: not run",
        "- Decisions: see DECISIONS.md; metric definitions: docs/metrics.md; preregistered thresholds: PLAN.md.\n",
        "## 2. Runs\n",
        _runs_table(),
        f"\nGPU-hours used (all runs): {gpu_hours_used():.2f}\n",
    ]
    if dev_tables and (dev_tables / "table.md").exists():
        parts += ["## 3. Development-set results (checkpoint selection, sampling freeze)\n", _read(dev_tables / "table.md")]
    if rl_tables and (rl_tables / "table.md").exists():
        parts += ["## 4. Post-training controls (dev)\n", _read(rl_tables / "table.md")]
    if final_tables and (final_tables / "table.md").exists():
        parts += ["## 5. Locked final evaluation (official validation shards; one pass per frozen method)\n", _read(final_tables / "table.md")]
    if figures:
        parts.append("## 6. Figures\n")
        for f in figures:
            parts.append(f"![{f.stem}]({f.relative_to(out.parent) if f.is_relative_to(out.parent) else f})\n")
    if anims:
        parts.append("## 7. Animations\n")
        for a in anims:
            parts.append(f"- `{a}`")
    parts += ["\n## 8. Limitations\n", _read(REPO_ROOT / "docs" / "limitations.md", "- See STATUS.md.")]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts) + "\n")
    pdf = out.with_suffix(".pdf")
    html = out.with_suffix(".html")
    try:
        subprocess.run(["pandoc", str(out), "-s", "-o", str(html), "--metadata", "title=Waymo_AV report"], check=True, cwd=out.parent)
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        if chrome:
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", f"--print-to-pdf={pdf}", "--no-pdf-header-footer", str(html)], check=True, timeout=120, capture_output=True)
            print("pdf:", pdf)
    except Exception as e:  # noqa
        print("PDF generation skipped:", e)
    print("report:", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-tables", default=None)
    ap.add_argument("--final-tables", default=None)
    ap.add_argument("--rl-tables", default=None)
    ap.add_argument("--figures", nargs="*", default=[])
    ap.add_argument("--anims", nargs="*", default=[])
    ap.add_argument("--out", default="reports/report.md")
    a = ap.parse_args(argv)
    build(Path(a.dev_tables) if a.dev_tables else None, Path(a.final_tables) if a.final_tables else None, Path(a.rl_tables) if a.rl_tables else None, [Path(f) for f in a.figures], [Path(x) for x in a.anims], REPO_ROOT / a.out)


if __name__ == "__main__":
    main()
