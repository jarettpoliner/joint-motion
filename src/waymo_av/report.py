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


def _demote(md: str, levels: int = 2) -> str:
    """Demote markdown headings so embedded comparison files nest under the report sections."""
    out = []
    for line in md.splitlines():
        if line.startswith("#"):
            n = len(line) - len(line.lstrip("#"))
            line = "#" * min(n + levels, 6) + line[n:]
        out.append(line)
    return "\n".join(out)


def _key_results(tables: Path, label: str) -> str:
    """Headline numbers generated from aggregate outputs (summary.json / paired_vs_reference.json)."""
    sj, pj = tables / "summary.json", tables / "paired_vs_reference.json"
    if not sj.exists():
        return f"- {label}: not available\n"
    S = json.load(open(sj)); P = json.load(open(pj)) if pj.exists() else {}
    fam = {"cv": "constant velocity", "mind_pilot": "independent decoder", "m0_pilot": "joint M0 (clean-trained)", "m1_pilot": "joint M1 (corruption-trained)", "select_m1": "M1 + best-6-of-32 reward selection", "extra_sft_pilot": "M1 + extra SFT", "filtered_sft_pilot": "M1 + reward-filtered SFT", "rl_pilot": "M1 + bounded RL"}
    rows = ["| method | clean joint minADE@6 (m) | core-suite joint minADE@6 (m) | clean overlap, own 6 outputs (%) | clean offroad (%) | n seeds |", "|---|---|---|---|---|---|"]
    core = ["missing_0.1", "missing_0.3", "missing_0.5", "noise_0.1", "noise_0.3", "noise_0.5"]
    for base, name in fam.items():
        ms = sorted({k.split("|")[0] for k in S if k.split("|")[0] == base or k.split("|")[0].startswith(base + "_s")})
        if not ms:
            continue
        g = lambda m, c, k: S.get(f"{m}|{c}", {}).get(k)
        cl = [g(m, "clean", "joint_minADE_8s") for m in ms if g(m, "clean", "joint_minADE_8s") is not None]
        cs = [sum(g(m, c, "joint_minADE_8s") for c in core) / len(core) for m in ms if all(g(m, c, "joint_minADE_8s") is not None for c in core)]
        ov = [g(m, "clean", "overlap_rate") for m in ms if g(m, "clean", "overlap_rate") is not None]
        of = [g(m, "clean", "offroad_frac") for m in ms if g(m, "clean", "offroad_frac") is not None]
        f = lambda v, p=2, sc=1: (f"{sc*sum(v)/len(v):.{p}f}" + (f" (seeds: {', '.join(f'{sc*x:.{p}f}' for x in v)})" if len(v) > 1 else "")) if v else "—"
        rows.append(f"| {name} | {f(cl)} | {f(cs)} | {f(ov, 1, 100)} | {f(of, 1, 100)} | {len(ms)} |")
    lines = [f"**{label}** (per-method means over seeds; CIs and paired differences in the tables below)", "", *rows, ""]
    ref = next((k for k in P if k.endswith("|CORE_SUITE") and k.startswith("m1_pilot")), None)
    if ref:
        d = P[ref]["joint_minADE_8s"]
        lines.append(f"- M1 vs M0 core-suite joint minADE@6 (seed of reference): {100*d['rel_delta']:+.1f} % [{100*d['rel_ci_low']:+.1f}, {100*d['rel_ci_high']:+.1f}] ({d.get('unit','scenario')}-level bootstrap). Per-seed verdicts against the preregistered thresholds: see the comparison section.")
    return "\n".join(lines) + "\n"


def _metric_qualification() -> str:
    out = []
    f = REPO_ROOT / "manifests" / "overlap_floor_audit.json"
    if f.exists():
        s = json.load(open(f))["summary"]
        out.append(f"- Logged-future overlap audit v2 (D-028a; {s['n_scenes']} dev scenes): reference rate on jointly valid intervals {100*s['rate_jointly_valid']:.1f} % ({s['n_flagged_jointly_valid']} scenes; classes {s['classes_of_jointly_valid_flags']}); the old unmasked full-horizon test flagged {s['n_flagged_unmasked_full_horizon']} scenes, {s['classes_of_unmasked_flags'].get('outside_jointly_valid', 0)} of them only on zero-filled invalid label steps. Superseded v1: manifests/overlap_floor_audit_v1_superseded.json.")
    for g in sorted((REPO_ROOT / "manifests").glob("model_overlap_inspection_*.json")):
        s = json.load(open(g))["summary"]
        out.append(f"- Generated-trajectory overlap inspection (D-028b; {s['tag']}, {s['condition']}): {s['n_flagged_pairs']}/{s['n_scene_sample_pairs']} (scene, sample) pairs flagged ({100*s['flag_rate_full_horizon']:.1f} %); verdicts {s['verdicts_all_flags']}; by stratum {s['verdicts_by_stratum']}. The RL overlap reward is an approximate geometric-consistency objective, not a verified safety measure.")
    f = REPO_ROOT / "manifests" / "overlap_depth_audit.json"
    if f.exists():
        s = json.load(open(f))
        out.append(f"- Penetration depth does not separate logged-reference flags from model flags (p50 {s['gt_depth_m']['p50']:.2f} m vs {s['model_depth_m']['p50']:.2f} m); no depth margin is used.")
    out.append("- Offroad: sign convention +1 calibrated by log playback (manifests/metric_config.json); KD-tree nearest-segment implementation verified against brute force (tests).")
    return "\n".join(out) + "\n"


def build(dev_tables: Path | None, final_tables: Path | None, rl_tables: Path | None, figures: list[Path], anims: list[Path], out: Path):
    paths = load_paths()
    cal = REPO_ROOT / "manifests" / "calibration.json"
    dv = REPO_ROOT / "manifests" / "dataset_version.json"
    parts = [
        "# Waymo_AV — Robust Joint Motion Forecasting Under Missing and Noisy Observations",
        f"\nGenerated {time.strftime('%Y-%m-%d %H:%M')} from commit `{git_commit()}`. Simplified MotionLM-style two-agent joint predictor; local metrics (not official challenge scores).\n",
        "## 0. Key results\n",
        _key_results(dev_tables, "Development set (500 scenes; checkpoint selection; NOT held-out)") if dev_tables else "",
        _key_results(final_tables, "FINAL held-out set (1,000 locked scenes; one pass per frozen method)") if final_tables and (final_tables / "summary.json").exists() else "- FINAL held-out results: pending\n",
        "RL outcome (D-031): recorded as mixed/negative under the frozen comparison rule — see section 3a (Q5/Q6) and DECISIONS.md.\n",
        "## 1. Setup\n",
        f"- Dataset: `{_read(dv, '{}').strip()}`" if dv.exists() else "- Dataset: NOT DOWNLOADED (blocked on licence login)",
        f"- Calibration: `{_read(cal, '{}').strip()[:1500]}`" if cal.exists() else "- Calibration: not run",
        "- Decisions: see DECISIONS.md; metric definitions: docs/metrics.md; preregistered thresholds: PLAN.md.\n",
        "## 2. Runs\n",
        _runs_table(),
        f"\nGPU-hours used (all runs): {gpu_hours_used():.2f}\n",
    ]
    parts += ["## 2b. Metric qualification (before any post-training)\n", _metric_qualification()]
    if dev_tables and (dev_tables / "table.md").exists():
        parts += ["## 3. Development-set results (checkpoint selection, sampling freeze)\n",
                  "Selection rules: M0/Mind checkpoints = lowest clean joint minADE@6 on the first 256 dev scenes; M1 = lowest mean of clean and core-suite joint minADE@6 on the same scenes (D-017). These selection scores are NOT comparable across models; every table below is a matched evaluation of the selected checkpoints on identical scenes, conditions and sampling.\n",
                  _read(dev_tables / "table.md")]
        if (dev_tables / "comparison.md").exists():
            parts += ["### 3a. Per-seed development comparison\n", _demote(_read(dev_tables / "comparison.md"))]
    if rl_tables and (rl_tables / "table.md").exists():
        parts += ["## 4. Post-training controls (dev)\n", _read(rl_tables / "table.md")]
    if final_tables and (final_tables / "table.md").exists():
        parts += ["## 5. Locked final evaluation (official validation shards; one pass per frozen method)\n", _read(final_tables / "table.md")]
        if (final_tables / "comparison.md").exists():
            parts += ["### 5a. Per-seed final comparison\n", _demote(_read(final_tables / "comparison.md"))]
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
