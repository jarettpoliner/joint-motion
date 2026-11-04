"""Build the README presentation assets in docs/media/ from existing, saved outputs only.

No model is loaded and nothing is evaluated. The script
  1. draws a simplified architecture diagram (content mirrors reports/figures/fig_architecture.* and
     src/waymo_av/model.py),
  2. crops / rearranges panels of the tracked FINAL figures in reports/figures/final/ so they stay
     legible at README width (pixels are copied, not re-plotted),
  3. palette-optimises two existing demo GIFs from reports/demos/ (same frames, frame timing and
     size; originals are left untouched; the MP4 versions are linked from the README),
and records every input (with sha256) and operation in docs/media/sources.json.

    <data_root>/env/train/bin/python scripts/build_readme_media.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"
FIN = ROOT / "reports" / "figures" / "final"
GIFS = {
    # README name: (source GIF, documented selection)
    "demo_noise0.5_M0_vs_M1_d42aba08c268ec8b.gif": (
        "reports/demos/prelim_M0_vs_M1_noise_0.5/d42aba08c268ec8b_noise_0.5.gif",
        "dev set; noise_0.5; rule 'mixed' by M1 joint minADE@6 over the first 64 dev-manifest scenes; selected_as=best"),
    "demo_clean_M1_selection_RL_d5d511761e2f9612.gif": (
        "reports/demos/final_M1_vs_selection_vs_RL_clean/d5d511761e2f9612_clean.gif",
        "dev set; clean; rule 'mixed' by M1_bounded_RL joint minADE@6 over the first 64 dev-manifest scenes; selected_as=worst"),
}
A_COL, B_COL = "#1f77b4", "#ff7f0e"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def architecture(out: Path) -> None:
    fig = plt.figure(figsize=(15, 6.6), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    boxes = [
        ("Observed history", "1 s at 10 Hz\n2 target agents\n+ 30 nearby agents\nmap, traffic lights", "#eef3fb", "#1f4e79"),
        ("Corruption", "history only: missing\nframes, position noise\nM1 trains on clean\n+ corrupted inputs", "#fdf0e6", "#a85a00"),
        ("Scene encoder", "early fusion of all\ninput tokens\n→ 64 latent queries\n2 layers, d = 128", "#eef3fb", "#1f4e79"),
        ("Joint decoder", "autoregressive\ntransformer, 4 layers\n169-token vocabulary\n2 agents × 16 steps", "#ebf5ea", "#2d6a2d"),
        ("Joint futures", "6 joint samples\nper scene; each one\nholds both agents'\n8 s trajectories", "#f2ebf7", "#5b2d82"),
    ]
    w, h, gap, y0 = 2.62, 2.2, 0.37, 4.2
    x = (15 - (5 * w + 4 * gap)) / 2
    for i, (title, body, fc, ec) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=2))
        ax.text(x + w / 2, y0 + h - 0.33, title, ha="center", va="center", fontsize=16.5, fontweight="bold", color=ec)
        ax.text(x + w / 2, y0 + h - 0.68, body, ha="center", va="top", fontsize=14, color="#222222", linespacing=1.4)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, y0 + h / 2), xytext=(x + w + 0.03, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#333333", mutation_scale=20))
        x += w + gap

    # timeline strip: 1 s observed history, then 16 time-major joint token steps at 2 Hz
    rng = np.random.default_rng(3)
    yA, yB = 2.55, 1.85
    hx = np.linspace(0.75, 3.25, 11)
    missing = {3, 7}
    for row_y, col in ((yA, A_COL), (yB, B_COL)):
        for k, xx in enumerate(hx):
            if k in missing:
                continue
            ax.plot(xx + rng.normal(0, 0.03), row_y + rng.normal(0, 0.06), "o", color=col, ms=8)
    ax.text(0.4, yA, "A", ha="center", va="center", fontsize=15, fontweight="bold", color=A_COL)
    ax.text(0.4, yB, "B", ha="center", va="center", fontsize=15, fontweight="bold", color=B_COL)
    ax.text(2.0, 3.35, "observed history (input)", ha="center", va="center", fontsize=14, color="#333333")
    ax.text(2.0, 1.3, "gaps, jitter = corruption", ha="center", va="center", fontsize=12, color="#a85a00")
    ax.text(0.75, 0.95, "−1 s", ha="center", va="center", fontsize=13, color="#555555")
    ax.plot([3.65, 3.65], [1.45, 3.15], "--", color="#555555", lw=1.5)
    ax.text(3.65, 0.95, "now", ha="center", va="center", fontsize=13, color="#555555")

    x0, x1, n = 4.05, 14.75, 16
    cw = (x1 - x0) / n
    s = 0.52
    hi = 5  # highlighted step (0-based): its tokens are predicted from steps 0..hi-1 of BOTH agents
    for t in range(n):
        cx = x0 + t * cw + (cw - s) / 2
        for row_y, col, lab in ((yA, A_COL, "A"), (yB, B_COL, "B")):
            done = t < hi
            fc = col if done else ("white" if t > hi else "#ffffff")
            ax.add_patch(Rectangle((cx, row_y - s / 2), s, s, fc=fc, ec=col, lw=2.6 if t == hi else 1.2, alpha=0.85 if done else 1.0))
            ax.text(cx + s / 2, row_y, f"{lab}{t + 1}", ha="center", va="center", fontsize=10.5,
                    color="white" if done else col, fontweight="bold" if t == hi else "normal")
        if t + 1 in (1, 4, 8, 12, 16):
            ax.text(cx + s / 2, 0.95, f"{(t + 1) / 2:g} s", ha="center", va="center", fontsize=13, color="#555555")
    ax.text((x0 + x1) / 2, 3.35, "future: motion tokens generated jointly, one 0.5 s step at a time", ha="center", va="center", fontsize=14, color="#333333")
    # bracket over the already-generated steps, arrow to the highlighted step
    bx0, bx1, by = x0 + 0.08, x0 + hi * cw - 0.08, 2.95
    ax.plot([bx0, bx0, bx1, bx1], [by - 0.08, by, by, by - 0.08], color="#2d6a2d", lw=1.8)
    ax.annotate("", xy=(x0 + hi * cw + cw / 2, yA + s / 2 + 0.04), xytext=(bx1, by),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#2d6a2d", connectionstyle="arc3,rad=-0.45", shrinkA=0, mutation_scale=16))
    ax.text(7.6, 0.5, "Step 6 tokens for A and B are both predicted from A1–A5 and B1–B5 (and the scene).", ha="center", va="center", fontsize=14, color="#2d6a2d")
    ax.text(7.6, 0.12, "The independent-decoder ablation lets each agent see only its own earlier tokens.", ha="center", va="center", fontsize=13, color="#555555")
    fig.savefig(out, dpi=160, facecolor="white")
    plt.close(fig)


def _body_gaps(a: np.ndarray, y0: int) -> np.ndarray:
    return ~(a[y0:] < 245).any(0)  # True = column is background in the panel rows


def rl_montage(src: Path, out: Path) -> dict:
    a = np.asarray(Image.open(src).convert("L"))
    img = Image.open(src).convert("RGB")
    H, W = a.shape
    dark = (a[200:990] < 100).sum(0)
    cols = [x for x in range(W) if dark[x] > 600]
    # axis spines are evenly spaced (~619 px); tall error bars inside a panel are not
    spines = [cols[0]]
    for x in cols:
        if 580 < x - spines[-1] < 660:
            spines.append(x)
    assert len(spines) == 7, spines
    # Split just LEFT of each panel's y tick labels: walk left from the spine over the tick marks,
    # the tick/label gap and the label glyphs (internal gaps <= 8 px, e.g. "17.5"), stop at the first
    # wider white run, and cut inside that run on a column that is blank in every row.
    plot_ink = (a[190:972] < 200).any(0)  # rows above the bottom axis line (y = 977-979)
    # blank in every row except the bottom axis lines (cutting a plain axis line only shortens it)
    blank_all = ~((a[95:974] < 245).any(0) | (a[983:] < 245).any(0))
    bounds = [0]
    for s in spines[1:]:
        x = s - 1
        while plot_ink[x]:          # tick marks
            x -= 1
        while not plot_ink[x]:      # gap between tick marks and labels
            x -= 1
        while True:                 # label glyphs; x is on glyph ink here
            while plot_ink[x]:
                x -= 1
            run = 0
            while not plot_ink[x - run]:
                run += 1
            if run > 8:
                label_left = x + 1
                break
            x -= run
        cands = [c for c in range(label_left - 4, label_left - 4 - 40, -1) if blank_all[c]]
        assert cands, f"no blank column left of the labels of the panel at spine {s}"
        bounds.append(cands[0])
    bounds.append(W)
    top = 92    # below the long figure suptitle (moved into the README caption)
    split = 186  # white rows between the panel titles (y < 176) and the axes (y > 190)
    # Panel titles can be wider than their axes, so the title rows get their own cut: the centre of
    # the widest blank run between consecutive spines.
    t_blank = ~(a[top:split] < 245).any(0)
    tbounds = [0]
    for left, right in zip(spines[:-1], spines[1:]):
        best, run = (0, 0), 0
        for x in range(left + 100, right):
            run = run + 1 if t_blank[x] else 0
            if run > best[0]:
                best = (run, x - run + 1)
        tbounds.append(best[1] + best[0] // 2)
    tbounds.append(W)
    panels = []
    for i in range(7):
        x0, x1 = min(bounds[i], tbounds[i]), max(bounds[i + 1], tbounds[i + 1])
        body = img.crop((bounds[i], split, bounds[i + 1], H))
        if i > 0:  # blank the previous panel's bottom axis line where it runs into this panel's labels
            px = body.load()
            # an axis-line pixel is dark on the line rows but blank just above and below it; vertical
            # glyph strokes of this panel's "0" tick label are dark there too, so they are kept
            for x in range(0, spines[i] - 30 - bounds[i]):
                gx = bounds[i] + x
                if a[970, gx] >= 200 and a[986, gx] >= 200:
                    for yy in range(975, 982):
                        if a[yy, gx] < 245:
                            px[x, yy - split] = (255, 255, 255)
        panel = Image.new("RGB", (x1 - x0, H - top), "white")
        panel.paste(img.crop((tbounds[i], top, tbounds[i + 1], split)), (tbounds[i] - x0, 0))
        panel.paste(body, (bounds[i] - x0, split - top))
        panels.append(panel)
    cw = max(p.width for p in panels)
    ch = max(p.height for p in panels)
    font_path = font_manager.findfont("DejaVu Sans")
    bold_path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    f_head, f_key = ImageFont.truetype(bold_path, 44), ImageFont.truetype(font_path, 33)
    head_h, pad = 78, 24
    canvas = Image.new("RGB", (4 * cw + 2 * pad, 2 * (ch + head_h) + 2 * pad), "white")
    d = ImageDraw.Draw(canvas)
    rows = [("Accuracy and sample spread", [0, 1, 6, None]),
            ("Geometric consistency (approximate tests; lower = fewer flags)", [2, 3, 4, 5])]
    for r, (head, idx) in enumerate(rows):
        y = pad + r * (ch + head_h)
        d.text((pad + 10, y + 12), head, font=f_head, fill="#1e2761")
        for c, i in enumerate(idx):
            x = pad + c * cw
            if i is None:
                key = ["M1  = corruption-trained joint model",
                       "M1+sel  = best 6 of 32 samples by reward",
                       "M1+xSFT = extra supervised steps",
                       "M1+fSFT = reward-filtered SFT",
                       "M1+RL  = bounded RL post-training",
                       "",
                       "Bars: mean of 3 seeds, 95 % CI",
                       "Dots: individual seeds",
                       "Final held-out set, 1,000 scenes,",
                       "clean inputs"]
                for k, line in enumerate(key):
                    d.text((x + 40, y + head_h + 90 + k * 48), line, font=f_key, fill="#222222")
                continue
            p = panels[i]
            canvas.paste(p, (x + (cw - p.width) // 2, y + head_h))
    canvas.save(out, optimize=True)
    return {"panel_bounds_px": bounds, "title_bounds_px": tbounds, "spines_px": spines, "crop_top_px": top, "title_axes_split_px": split,
            "layout": "row 1: clean jminADE, core-suite jminADE, diversity, key; row 2: overlap own 6, overlap 32-pool, offroad, kinematically implausible"}


def severity_crop(src: Path, out: Path) -> dict:
    a = np.asarray(Image.open(src).convert("L"))
    H, W = a.shape
    top = 76  # below the long figure suptitle (moved into the README caption)
    ink = (a[top:] < 245).any(0)
    xs = np.where(ink)[0]
    box = (max(int(xs[0]) - 30, 0), top, min(int(xs[-1]) + 30, W), H)
    Image.open(src).convert("RGB").crop(box).save(out, optimize=True)
    return {"crop_box_px": box}


def optimise_gif(src: Path, out: Path) -> dict:
    import imageio_ffmpeg

    vf = "split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=none:diff_mode=rectangle"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(src), "-vf", vf, "-loop", "0", str(out)], check=True)
    a, b = Image.open(src), Image.open(out)
    assert (a.n_frames, a.size) == (b.n_frames, b.size), "frame count or size changed"
    return {"ffmpeg_filter": vf, "frames": b.n_frames, "size_px": list(b.size), "bytes_before": src.stat().st_size, "bytes_after": out.stat().st_size}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"note": "README presentation assets derived from saved outputs; no model run or evaluation.", "assets": {}}
    architecture(OUT / "architecture.png")
    rec["assets"]["architecture.png"] = {"op": "drawn by scripts/build_readme_media.py; content mirrors reports/figures/fig_architecture.svg and src/waymo_av/model.py"}
    for name, src, fn in (("robustness_vs_severity_final.png", FIN / "fig_error_vs_severity.png", severity_crop),
                          ("rl_controls_final.png", FIN / "fig_rl_controls.png", rl_montage)):
        info = fn(src, OUT / name)
        rec["assets"][name] = {"source": str(src.relative_to(ROOT)), "source_sha256": sha(src), "op": fn.__name__, **info,
                               "source_data": json.loads((src.with_suffix("").with_suffix(".sources.json")).read_text()).get("eval_dir")
                               if src.with_suffix("").with_suffix(".sources.json").exists() else None}
    for name, (src, sel) in GIFS.items():
        s = ROOT / src
        info = optimise_gif(s, OUT / name)
        demo_dir = s.parent
        rec["assets"][name] = {"source": src, "source_sha256": sha(s), "mp4": str(s.with_suffix(".mp4").relative_to(ROOT)),
                               "demo_metrics": str((demo_dir / "demo_metrics.json").relative_to(ROOT)), "selection": sel, "op": "optimise_gif", **info}
    (OUT / "sources.json").write_text(json.dumps(rec, indent=1))
    for p in sorted(OUT.iterdir()):
        print(f"{p.stat().st_size / 1024:8.0f} KB  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
