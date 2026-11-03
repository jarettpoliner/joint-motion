# Slides

`waymo_av_results_prelim.pptx` / `.pdf` — PRELIMINARY deck (development set, seed 0). Regenerate the
content JSON from saved results, then build:

```
PY=<data_root>/env/train/bin/python
PYTHONPATH=src $PY scripts/build_slide_content.py --dev-eval <data_root>/runs/eval_dev --seeds 0,1,2 \
    --label "PRELIMINARY (development set)" --figures reports/figures/dev \
    --demos reports/demos/<set1> reports/demos/<set2> [--final-eval <data_root>/runs/eval_final]
NODE_PATH=$PWD/slides/node_modules PPTX_SKILL_DIR=<pptx skill dir> node slides/build_deck.js slides/content.json slides/<out>.pptx
```
`slides/content.json` records every number, figure path, checkpoint id and the regeneration command used.
Final deck: same command with `--final-eval` after the freeze and locked final evaluation.
