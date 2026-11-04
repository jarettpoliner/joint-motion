// Build the Waymo_AV results deck from slides/content.json (saved results only).
// node slides/build_deck.js [content.json] [out.pptx]
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL_DIR + "/scripts/apply_theme.js");

const contentPath = process.argv[2] || path.join(__dirname, "content.json");
const outPath = process.argv[3] || path.join(__dirname, "waymo_av_results.pptx");
const C0 = JSON.parse(fs.readFileSync(contentPath, "utf8"));
const LABEL = C0.label;
const THEME = { name: "WaymoAV", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1A1A1A", lt1: "FFFFFF", dk2: "1E2761", lt2: "EEF2F8", accent1: "1E2761", accent2: "D62728", accent3: "1F77B4", accent4: "9467BD", accent5: "2CA02C", accent6: "FF7F0E", hlink: "1E2761", folHlink: "9467BD" } };

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.author = "Waymo_AV project"; pres.title = "Robust Joint Motion Forecasting Under Missing and Noisy Observations";
  const C = pres.SchemeColor;
  const footer = `Waymo_AV · commit ${C0.commit} · ${C0.date} · ${LABEL}`;
  pres.defineSlideMaster({ title: "DARK", background: { color: THEME.colors.dk2 },
    objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.7, y: 2.2, w: 11.9, h: 1.6, fontSize: 40, bold: true, color: C.background1, align: "left", margin: 0 } } },
              { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.9, w: 11.9, h: 2.4, fontSize: 18, color: "CADCFC", align: "left", margin: 0 } } },
              { text: { text: footer, options: { x: 0.7, y: 6.9, w: 11.9, h: 0.35, fontSize: 10, color: "CADCFC", margin: 0 } } }] });
  pres.defineSlideMaster({ title: "DARK_CONTENT", background: { color: THEME.colors.dk2 },
    objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.35, w: 11.9, h: 0.9, fontSize: 30, bold: true, color: C.background1, align: "left", margin: 0 } } },
              { text: { text: footer, options: { x: 0.7, y: 6.9, w: 11.9, h: 0.35, fontSize: 10, color: "CADCFC", margin: 0 } } }] });
  pres.defineSlideMaster({ title: "LIGHT", background: { color: THEME.colors.lt1 },
    objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontSize: 30, bold: true, color: C.text2, align: "left", margin: 0 } } },
              { text: { text: footer, options: { x: 0.6, y: 7.0, w: 8.6, h: 0.3, fontSize: 9, color: "777777", margin: 0 } } },
              { placeholder: { options: { name: "tag", type: "body", x: 9.3, y: 7.0, w: 3.4, h: 0.3, fontSize: 9, bold: true, color: C.accent2, align: "right", margin: 0 } } }] });
  const TAG = LABEL.split(" (")[0];  // short form for the corner tag; the footer carries the full label
  const addLight = (title, tag) => { const s = pres.addSlide({ masterName: "LIGHT" }); s.addText(title, { placeholder: "title" }); s.addText(tag || TAG, { placeholder: "tag" }); return s; };
  const bullets = (s, items, o) => s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })), Object.assign({ isTextBox: true, fontSize: 14, color: C.text1, valign: "top", margin: 0 }, o));
  const card = (s, x, y, w, h, head, body, fill, bodySize) => { s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill || THEME.colors.lt2 }, line: { color: THEME.colors.lt2 }, rectRadius: 0.08 }); s.addText(head, { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: 0.4, fontSize: 15, bold: true, color: C.text2, isTextBox: true, margin: 0 }); s.addText(body, { x: x + 0.15, y: y + 0.55, w: w - 0.3, h: h - 0.65, fontSize: bodySize || 12, color: C.text1, isTextBox: true, valign: "top", margin: 0 }); };
  const stat = (s, x, y, w, big, small) => { s.addText(big, { x, y, w, h: 0.75, fontSize: 32, bold: true, color: C.text2, isTextBox: true, align: "center", margin: 0 }); s.addText(small, { x, y: y + 0.75, w, h: 0.5, fontSize: 11, color: "555555", isTextBox: true, align: "center", margin: 0 }); };
  const img = (s, p, o) => { if (p && fs.existsSync(p)) s.addImage(Object.assign({ path: p }, o)); else s.addText("PENDING — figure not yet generated", Object.assign({ isTextBox: true, fontSize: 14, color: C.accent2, align: "center", valign: "middle", fill: { color: THEME.colors.lt2 } }, o)); };
  const dev = C0.dev, fin = C0.final;
  const R = (blk, path) => path.split(".").reduce((o, k) => (o && o[k] !== undefined ? o[k] : null), blk);
  const val = (blk, p) => (blk && blk.available ? (R(blk, p) || "pending") : "pending");

  // 1 title
  let s = pres.addSlide({ masterName: "DARK" });
  s.addText("Robust Joint Motion Forecasting Under Missing and Noisy Observations", { placeholder: "title" });
  s.addText(`Compact MotionLM-style joint forecaster on the Waymo Open Motion Dataset — corruption-aware training and bounded RL post-training\n\nStatus: ${LABEL}. Development numbers use manifests/dev.json; held-out numbers use the locked manifests/final.json${fin && fin.available ? "" : " (final evaluation pending)"}.`, { placeholder: "body" });
  s.addNotes("Deck generated from saved results only: slides/content.json, built by scripts/build_slide_content.py and slides/build_deck.js. Regenerate: " + C0.regenerate);

  // 2 research question
  s = addLight("Research question and motivation");
  card(s, 0.6, 1.4, 3.9, 3.2, "Question", "Does a joint two-agent autoregressive forecaster keep its accuracy and physical consistency when the observed history is incomplete or noisy — and can training on corrupted observations, plus bounded reward post-training, make it more robust without hurting clean accuracy?");
  card(s, 4.7, 1.4, 3.9, 3.2, "Why it matters", "Deployed trackers drop frames and jitter. Forecasters trained only on clean logs can degrade sharply; robustness must be measured with the same evaluator on the same scenes, with uncertainty, before claiming an improvement.");
  card(s, 8.8, 1.4, 3.9, 3.2, "Approach", "Reproduce a compact MotionLM-style joint model (M0), train a matched corruption-aware variant (M1) and an independent-agent ablation, then post-train M1 with REINFORCE + KL + replay against simpler controls. Preregistered thresholds; paired group bootstrap.");
  bullets(s, ["Preregistered success rule (PLAN.md): ≥ 5 % relative reduction of core-suite joint minADE@6, ≤ 3 % clean worsening, ≤ 0.5 pp overlap increase", "Scope kept to forecasting, robustness and bounded RL — no observation selection, no architecture expansion"], { x: 0.6, y: 4.9, w: 12.1, h: 1.6 });

  // 3 task & dataset
  s = addLight("Task and dataset");
  stat(s, 0.6, 1.3, 2.4, `${C0.dataset.train.toLocaleString()}`, "training windows (9,144 after removing scenes that share a source segment with dev)");
  stat(s, 3.1, 1.3, 2.4, `${C0.dataset.dev}`, "development scenes (selection, tuning, sampling freeze)");
  stat(s, 5.6, 1.3, 2.4, `${C0.dataset.final.toLocaleString()}`, `final held-out scenes, ${C0.dataset.final_locked ? "LOCKED" : "not locked"} (official validation shards)`);
  stat(s, 8.1, 1.3, 2.4, "1 s → 8 s", "history @ 10 Hz → 16 joint steps @ 2 Hz");
  stat(s, 10.6, 1.3, 2.2, "2", "interacting targets per scene (objects_of_interest)");
  img(s, C0.figures.fig_real_scenes, { x: 0.6, y: 2.9, w: 12.1, h: 3.15 });
  bullets(s, [`${C0.dataset.version}, ${C0.dataset.shards}; ${C0.dataset.excluded_timing_pct} % of pair scenarios excluded by the 10 Hz timing rule (dropped frames, not interpolated)`, `Calibration on dev: tokenizer saturation ${C0.calibration.tokenizer_saturation_pct} %, reconstruction ${C0.calibration.recon_err_m} m; offroad convention validated by log playback (${C0.calibration.oracle_offroad_pct} % offroad); logged-future overlap reference ${C0.calibration.gt_overlap_ref_pct ?? "pending"} % on jointly valid intervals (audit v2)`], { x: 0.6, y: 6.1, w: 12.1, h: 0.8, fontSize: 11 });

  // 4 architecture
  s = addLight("Implemented architecture");
  img(s, C0.figures.fig_architecture, { x: 0.6, y: 1.25, w: 12.1, h: 5.6 });

  // 5 experimental design
  s = addLight("Experimental design and baselines");
  const rows = [["Method", "Training observations", "Role"].map(t => ({ text: t, options: { bold: true, fill: { color: THEME.colors.lt2 } } })),
    ["Constant position / constant velocity", "— (deterministic, 1 sample)", "non-learned floor; CV uses the model's own d₀ estimator"],
    ["Independent decoder (Mind)", "clean", "ablation: same model, no cross-agent attention"],
    ["Joint M0", "clean", "clean-trained baseline (reference for all paired deltas)"],
    ["Joint M1", "50 % clean + core corruption suite", "corruption-aware model"],
    ["M1 + extra SFT", "as M1, matched extra steps", "control: more supervised training"],
    ["M1 + reward-filtered SFT", "own samples, reward-filtered", "control: reward signal without policy gradient"],
    ["M1 + bounded RL", "REINFORCE, K = 8, LOO baseline, KL, replay", "post-training (reward: overlap, offroad, kinematics)"]];
  s.addTable(rows, { x: 0.6, y: 1.3, w: 7.6, colW: [2.4, 2.6, 2.6], fontSize: 10.5, color: THEME.colors.dk1, border: { type: "solid", color: "DDDDDD", pt: 0.5 }, rowH: 0.42 });
  card(s, 8.5, 1.3, 4.2, 2.5, "Corruption suite (observations only)", "Core: missing 10/30/50 % of history frames; Gaussian position noise σ = 0.1/0.3/0.5 m.\nHeld-out severities: missing 70 %, σ = 1.0 m, compound.\nThe current anchor frame is kept; ground truth is never corrupted.");
  card(s, 8.5, 4.0, 4.2, 2.6, "Measurement", "Joint minADE@6 / minFDE@6 over the same sample index (6 samples; 32-sample audit for overlap/diversity).\nOBB overlap at 10 Hz, road-edge offroad (vehicles), kinematic plausibility, diversity.\nPaired bootstrap over heuristic source-segment groups; 3 seeds; checkpoints chosen on dev only; final set evaluated once after freeze.");

  // 6 clean results
  const blk = (fin && fin.available) ? fin : dev;
  const blkName = (fin && fin.available) ? "FINAL held-out set (1,000 locked scenes)" : `development set (${dev && dev.available ? dev.n_scenes : "—"} scenes) — PRELIMINARY`;
  s = addLight("Clean forecasting results", (fin && fin.available) ? "FINAL" : TAG);
  img(s, C0.figures.fig_baseline_comparison, { x: 0.6, y: 1.25, w: 7.9, h: 2.9 });
  card(s, 8.7, 1.25, 4.0, 2.9, "Joint minADE@6 (m), clean, " + (blk && blk.available ? `seeds ${blk.seeds.join(",")}` : "pending"),
    `Constant velocity: ${val(blk, "clean.cv.jminADE")}\nIndependent: ${val(blk, "clean.mind_pilot.jminADE")}\nJoint M0: ${val(blk, "clean.m0_pilot.jminADE")}\nJoint M1: ${val(blk, "clean.m1_pilot.jminADE")}\nJoint − independent: ${val(blk, "deltas.joint_vs_independent_clean")}`);
  bullets(s, [`Evaluated on the ${blkName}; identical scenes and sampling (6 joint samples, T = 1.0, top-p 0.95) for every learned model; cp/cv are deterministic`, `Overlap, clean, 32-sample audit: M0 ${val(blk, "clean.m0_pilot.overlap32_pct")} over the full generated horizon; on jointly valid intervals M0 ${val(blk, "clean.m0_pilot.overlap32_jv_pct")} vs logged reference ${val(blk, "clean.oracle_log.overlap32_jv_pct")} (same approximate test applied to logs; not a lower bound)`, `Checkpoints selected on dev by their own criterion (M0: clean jminADE; M1: mean of clean and suite); 12k steps, 1.79 M parameters, d = 128`], { x: 0.6, y: 4.4, w: 12.1, h: 2.4 });

  // 7 robustness
  s = addLight("Robustness to missing and noisy observations", (fin && fin.available) ? "FINAL" : TAG);
  img(s, C0.figures.fig_error_vs_severity, { x: 0.6, y: 1.25, w: 7.6, h: 2.9 });
  img(s, C0.figures.fig_tradeoff, { x: 8.4, y: 1.25, w: 4.3, h: 2.9 });
  const vrows = [["seed", "core-suite rel. Δ (≤ −5 %)", "clean rel. Δ (≤ +3 %)", "core overlap Δ pp (≤ +0.5)", "verdict"].map(t => ({ text: t, options: { bold: true, fill: { color: THEME.colors.lt2 } } }))];
  (blk && blk.available && blk.verdicts ? blk.verdicts : []).forEach(v => vrows.push([String(v.seed), v.core_rel, v.clean_rel, v.overlap_pp, { text: v.verdict, options: { bold: true, color: v.verdict === "PASS" ? THEME.colors.accent5 : THEME.colors.accent2 } }]));
  if (vrows.length === 1) vrows.push(["—", "pending", "pending", "pending", "pending"]);
  s.addTable(vrows, { x: 0.6, y: 4.35, w: 7.6, colW: [0.7, 2.1, 2.0, 2.0, 0.8], fontSize: 10.5, border: { type: "solid", color: "DDDDDD", pt: 0.5 }, rowH: 0.34 });
  card(s, 8.4, 4.35, 4.3, 2.4, "M1 vs M0 (paired, group bootstrap)", `Core suite jminADE: ${val(blk, "deltas.m1_vs_m0_core")}\nHeld-out suite: ${val(blk, "deltas.m1_vs_m0_held")}\nClean: ${val(blk, "deltas.m1_vs_m0_clean")}\nCore overlap: ${val(blk, "deltas.m1_vs_m0_core_overlap_pp")} pp`);

  // 8 RL / controls
  s = addLight("Bounded RL post-training and controls", (fin && fin.available) ? "FINAL" : TAG);
  img(s, C0.figures.fig_rl_controls, { x: 0.6, y: 1.25, w: 12.1, h: 3.0 });
  const rlAvail = blk && blk.available && blk.methods && blk.methods.some(m => m.startsWith("rl_pilot"));
  // Short lines (< 60 characters) so the card keeps a margin when Calibri is substituted by a wider font.
  const rlv = (blk && blk.rl_verdicts) || [];
  const rlRule = !rlv.length ? "pending" : (rlv.every(v => v.verdict === rlv[0].verdict) ? `${rlv[0].verdict.toLowerCase()} in seeds ${rlv.map(v => v.seed).join(", ")}` : rlv.map(v => "seed " + v.seed + " " + v.verdict.toLowerCase()).join("; "));
  card(s, 0.6, 4.4, 6.0, 2.45, "RL vs its M1 initialisation (clean, paired)", rlAvail ? [
    `Clean jminADE@6: ${val(blk, "deltas.rl_vs_m1_clean")}`,
    `Overlap, 32 samples: ${val(blk, "deltas.rl_vs_m1_overlap32_pp")} pp`,
    `Offroad: ${val(blk, "deltas.rl_vs_m1_offroad_pp")} pp`,
    `Diversity, pairwise FDE: ${val(blk, "deltas.rl_vs_m1_diversity")} m`,
    `Excessive stopping: RL ${(blk.stopping||{}).rl_pilot ?? "—"} vs M1 ${(blk.stopping||{}).m1_pilot ?? "—"} (dev diagnostic)`,
    `   ends < 1 m when logged motion > 5 m, per agent-sample`,
    `Frozen rule: ${rlRule}`,
    `   needs: beat both SFT controls on overlap, clean Δ ≤ +3 %`].join("\n") : "PENDING — post-training stage not yet run. Runs planned: rl_pilot, filtered_sft_pilot, extra_sft_pilot per seed, all initialised from the dev-selected M1 checkpoint.", undefined, 11);
  card(s, 6.8, 4.4, 5.9, 2.45, "Controls vs M1", rlAvail ? [
    `Reward-filtered SFT, overlap32: ${val(blk, "deltas.fsft_vs_m1_overlap32_pp")} pp`,
    `Extra SFT (matched steps), overlap32: ${val(blk, "deltas.esft_vs_m1_overlap32_pp")} pp`,
    `Reward-based selection, best 6 of 32:`,
    `   own-output overlap ${val(blk, "deltas.select_vs_m1_overlap6_pp")} pp`,
    `   clean jminADE ${val(blk, "deltas.select_vs_m1_clean")}`,
    `   cost: 32 rollouts + reward scoring per scene`,
    `Reward = approximate geometric consistency, not verified safety`].join("\n") : "PENDING — controls run in the same stage. Interpretation rule (frozen before final evaluation): an RL effect counts only if it exceeds both the extra-SFT and reward-filtered-SFT controls under identical sampling.", undefined, 11);

  // 9 qualitative
  s = addLight("Qualitative examples and failures", LABEL);
  const demos = C0.demos || [];
  if (demos.length === 0) { s.addText("PENDING — animations not yet generated (python -m waymo_av.demo …)", { x: 0.6, y: 1.3, w: 12.1, h: 4.5, isTextBox: true, fontSize: 16, color: C.accent2, align: "center", valign: "middle", fill: { color: THEME.colors.lt2 } }); }
  else {
    const pick = [demos.find(d => d.selected_as === "best") || demos[0], demos.find(d => d.selected_as === "worst") || demos[demos.length - 1]];
    pick.forEach((d, i) => { img(s, d.still, { x: 0.6 + i * 6.15, y: 1.25, w: 5.95, h: 2.25 }); s.addText(`${d.selected_as === "best" ? "Success" : "Failure"} case — scenario ${d.scenario_id}, condition ${d.condition} — ${Object.entries(d.scores).map(([k, v]) => k.replace("jminADE_8s_", "") + " " + Number(v).toFixed(2) + " m").join(", ")}`, { x: 0.6 + i * 6.15, y: 3.55, w: 5.95, h: 0.55, isTextBox: true, fontSize: 10, color: "444444", margin: 0 }); });
    bullets(s, [`Selection rule (not hand-picked): ${demos[0].rule}`, "Each panel shows the identical scene, corruption realisation and sampling; dots = observed history as given to the model, black dashed = recorded future (never an input), coloured = generated joint samples with one shared sample index for both agents", "Animations (GIF + MP4, synchronized, fixed camera): reports/demos/<set>/<scenario>_<condition>.{gif,mp4}"], { x: 0.6, y: 4.25, w: 12.1, h: 2.5, fontSize: 12 });
  }

  // 10 conclusions
  s = pres.addSlide({ masterName: "DARK_CONTENT" });
  s.addText("Conclusions, limitations, remaining work", { placeholder: "title" });
  const concl = [];
  if (blk && blk.available) { concl.push(`Clean: joint M0 ${val(blk, "clean.m0_pilot.jminADE")} m vs CV ${val(blk, "clean.cv.jminADE")} m; joint − independent ${val(blk, "deltas.joint_vs_independent_clean")}`); concl.push(`Robustness: M1 − M0 on the core suite ${val(blk, "deltas.m1_vs_m0_core")}; clean ${val(blk, "deltas.m1_vs_m0_clean")}; verdicts: ${(blk.verdicts || []).map(v => "seed " + v.seed + " " + v.verdict).join(", ") || "pending"}`); }
  else concl.push("Results pending");
  concl.push(rlAvail ? `RL (mixed/negative): overlap ${val(blk, "deltas.rl_vs_m1_overlap32_pp")} pp vs M1 but clean jminADE ${val(blk, "deltas.rl_vs_m1_clean")}, diversity ${val(blk, "deltas.rl_vs_m1_diversity")} m, excessive stopping ${(blk.stopping||{}).rl_pilot ?? "—"} vs ${(blk.stopping||{}).m1_pilot ?? "—"}; frozen rule: ${(blk.rl_verdicts||[]).map(v => "seed " + v.seed + " " + v.verdict).join(", ") || "pending"}; the best-6-of-32 selection control dominates RL on every axis at 32-sample cost` : "RL and controls: pending");
  concl.push(`Limitations: compact model and ~9k training scenes (not a leaderboard reproduction); official metrics package unavailable (custom metrics audited on oracles); the 2 Hz box-overlap test is approximate (logged reference ${C0.calibration.gt_overlap_ref_pct ?? "—"} % on jointly valid intervals; RL reward = approximate geometric consistency, not verified safety); source-segment grouping is heuristic; synthetic corruptions are stress tests, not sensor models`);
  concl.push((fin && fin.available) ? "Final held-out evaluation done once per frozen method; see report for full tables" : (rlAvail ? "Remaining: freeze method choices, single locked final evaluation, final report and deck" : "Remaining: RL + controls (running), freeze method choices, single locked final evaluation, final report and deck"));
  s.addText(concl.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < concl.length - 1, paraSpaceAfter: 8 } })), { x: 0.7, y: 1.5, w: 11.9, h: 5.2, fontSize: 15, color: "FFFFFF", isTextBox: true, valign: "top", margin: 0 });
  s.addNotes("Regenerate: " + C0.regenerate + "\nGPU-hours used at build time: " + C0.gpu_hours);

  await pres.writeFile({ fileName: outPath });
  await applyTheme(outPath, THEME);
  console.log("wrote", outPath);
})();
