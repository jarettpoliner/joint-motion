"""Resumable campaign runner. Stages are skipped when their completion artifact exists and is
valid; failures stop the campaign visibly. State is written to registry/campaign_state.json.

python -m waymo_av.campaign --stage all|audit|download|preprocess|manifests|calibrate|overfit|pilot|dev_eval|freeze|posttrain|final_eval|report

Stage dependencies: download -> preprocess -> manifests -> calibrate -> overfit -> pilot -> dev_eval
-> freeze -> posttrain -> final_eval -> report. `final_eval` refuses to run unless `freeze` wrote
manifests/freeze.json (frozen checkpoints, sampling settings, thresholds).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from waymo_av.budget import check_disk, gpu_hours_used
from waymo_av.corruption import CORE_SUITE, HELDOUT_SUITE
from waymo_av.paths import REPO_ROOT, load_paths

STATE = REPO_ROOT / "registry" / "campaign_state.json"
P = load_paths()
DATA = Path(P["data_root"])
RUNS = Path(P["runs_dir"])
PY = P["train_python"]
GCLOUD = Path(P["gcloud_bin"])
CORE = ["clean"] + [s.name for s in CORE_SUITE]
HELDOUT = [s.name for s in HELDOUT_SUITE]


def _state():
    return json.load(open(STATE)) if STATE.exists() else {"stages": {}}


def _set(stage, status, **info):
    s = _state()
    s["stages"][stage] = {"status": status, "t": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
    STATE.parent.mkdir(exist_ok=True)
    json.dump(s, open(STATE, "w"), indent=1, default=str)
    print(f"[campaign] {stage}: {status} {info if info else ''}", flush=True)


def run(cmd: list[str] | str, **kw):
    print("[campaign] $", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=REPO_ROOT, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")


def gcloud_authenticated() -> bool:
    try:
        out = subprocess.check_output([str(GCLOUD / "gcloud"), "auth", "list", "--format=value(account)"], text=True, stderr=subprocess.STDOUT)
        return bool(out.strip())
    except Exception:
        return False


def _completed(run_name):
    p = REPO_ROOT / "registry" / "runs" / f"{run_name}.json"
    return p.exists() and json.load(open(p)).get("status") == "completed"


def _train(cfg, run_name=None, overrides=()):
    cmd = [PY, "-m", "waymo_av.train", "--config", cfg, "--resume"]
    if run_name:
        cmd += ["--run-name", run_name]
    if overrides:
        cmd += ["--override", *overrides]
    run(cmd)


# ---------------- stages ----------------
def stage_audit():
    info = {"gpu_hours_used": gpu_hours_used(), "disk": check_disk(0), "gcloud_auth": gcloud_authenticated()}
    nv = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout
    info["nvidia_smi"] = nv.strip()
    _set("audit", "completed", **info)
    return info


def stage_download(n_train=24, n_val=8, version="1_3_0"):
    if not gcloud_authenticated():
        _set("download", "blocked", reason="no gcloud credential; run: gcloud auth login (account that accepted the WOMD licence)")
        return False
    raw = DATA / "raw" / f"womd_v{version}" / "scenario"
    have_t = len(list((raw / "training").glob("*.tfrecord-*"))) if raw.exists() else 0
    have_v = len(list((raw / "validation").glob("*.tfrecord-*"))) if raw.exists() else 0
    if have_t >= n_train and have_v >= n_val and (REPO_ROOT / "manifests" / "dataset_version.json").exists():
        _set("download", "completed", skipped=True, train_shards=have_t, val_shards=have_v)
        return True
    run(["bash", "scripts/download_womd.sh", str(n_train), str(n_val), version])
    _set("download", "completed", train_shards=n_train, val_shards=n_val)
    return True


def stage_preprocess(n_train=24, n_val=8):
    run([PY, "-m", "waymo_av.data.preprocess", "--split", "training", "--shards", f"0-{n_train-1}"])
    run([PY, "-m", "waymo_av.data.preprocess", "--split", "validation", "--shards", f"0-{n_val-1}"])
    _set("preprocess", "completed")


def stage_manifests():
    f = REPO_ROOT / "manifests" / "final.json"
    if f.exists() and json.load(open(f)).get("locked"):
        _set("manifests", "completed", skipped=True)
        return
    run([PY, "-m", "waymo_av.data.manifests", "--lock-final"])
    _set("manifests", "completed")


def stage_calibrate():
    if (REPO_ROOT / "manifests" / "calibration.json").exists():
        _set("calibrate", "completed", skipped=True)
        return
    run([PY, "-m", "waymo_av.calibrate"])
    _set("calibrate", "completed")


def stage_overfit():
    if not _completed("m0_overfit"):
        _train("configs/train/m0_overfit.yaml")
    rec = json.load(open(REPO_ROOT / "registry" / "runs" / "m0_overfit.json"))
    _set("overfit", "completed", best=rec.get("best"))


def stage_pilot(seeds=(0,)):
    for s in seeds:
        for cfg, base in (("configs/train/m0_pilot.yaml", "m0_pilot"), ("configs/train/m1_pilot.yaml", "m1_pilot"), ("configs/train/mind_pilot.yaml", "mind_pilot")):
            name = f"{base}_s{s}"
            if not _completed(name):
                _train(cfg, run_name=name, overrides=[f"seed={s}", f"run_name={name}"])
    _set("pilot", "completed", seeds=list(seeds))


def _methods(seeds, which=("m0_pilot", "m1_pilot", "mind_pilot")):
    ms = ["cp", "cv", "oracle_log"]  # oracle_log = logged-future reference for overlap_rate_jointvalid / offroad
    for s in seeds:
        for base in which:
            ck = RUNS / f"{base}_s{s}" / "ckpt_best.pt"
            if ck.exists():
                ms.append(f"ckpt:{ck}@{base}_s{s}")
                if base == "m1_pilot":  # reward-based candidate-selection control on the same checkpoint
                    ms.append(f"select:{ck}@select_m1_s{s}")
    return ms


def stage_dev_eval(seeds=(0,)):
    out = RUNS / "eval_dev"
    if (out / "summary_all.json").exists():
        _set("dev_eval", "completed", skipped=True)
    else:
        run([PY, "-m", "waymo_av.evaluate", "--methods", *_methods(seeds), "--conditions", *CORE, *HELDOUT, "--manifest", "manifests/dev.json", "--out", str(out), "--tag", "dev"])
        _set("dev_eval", "completed", out=str(out))
    seeds_csv = ",".join(str(s) for s in seeds)
    run([PY, "-m", "waymo_av.aggregate", "--eval-dir", str(out), "--reference", f"m0_pilot_s{seeds[0]}", "--out", "reports/tables/dev"])
    run([PY, "-m", "waymo_av.compare", "--eval-dir", str(out), "--seeds", seeds_csv, "--out", "reports/tables/dev/comparison.md"])
    run([PY, "-m", "waymo_av.figures", "--eval-dir", str(out), "--seeds", seeds_csv, "--out", "reports/figures/dev", "--label", "development set (checkpoint selection; not held-out)"])


def stage_freeze(seeds=(0,)):
    """Freeze checkpoints, sampling settings and thresholds BEFORE any final-set evaluation."""
    fz = REPO_ROOT / "manifests" / "freeze.json"
    if fz.exists():
        _set("freeze", "completed", skipped=True)
        return
    methods = {}  # name -> evaluate.py method string (kind:path)
    for s in seeds:
        for base in ("m0_pilot", "m1_pilot", "mind_pilot"):
            ck = RUNS / f"{base}_s{s}" / "ckpt_best.pt"
            if ck.exists():
                methods[f"{base}_s{s}"] = f"ckpt:{ck}"
                if base == "m1_pilot":
                    methods[f"select_m1_s{s}"] = f"select:{ck}"
    for base in ("rl_pilot", "filtered_sft_pilot", "extra_sft_pilot"):
        for s in seeds:
            for fn in ("ckpt_best.pt", "ckpt_final.pt"):
                ck = RUNS / f"{base}_s{s}" / fn
                if ck.exists():
                    methods[f"{base}_s{s}"] = f"ckpt:{ck}"
                    break
    freeze = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "methods": methods,
        "sampling": {"n_samples": 6, "audit_samples": 32, "temperature": 1.0, "top_p": 0.95},
        "conditions": CORE + HELDOUT,
        "thresholds": {"primary_rel_reduction_core_suite": 0.05, "clean_guardrail_rel_worsening": 0.03, "overlap_guardrail_pp": 0.5},
        "comparison_rules": {
            "reference": "m0_pilot (clean-trained joint) per seed; paired, group bootstrap (D-025a)",
            "rl_effect_counts_only_if": "exceeds BOTH extra_sft and filtered_sft controls under identical sampling, on overlap_rate AND overlap_rate_jointvalid (32-sample audit), without clean jminADE worsening > 3 %",
            "reward_semantics": "approximate geometric-consistency objective (D-028a/b), not a verified safety measure",
            "selection_control": "select_m1_s*: best-6-of-32 by reward from the same M1 checkpoint (inference-time control)",
        },
        "final_manifest_sha256": json.load(open(REPO_ROOT / "manifests" / "final.json"))["sha256_ids"],
        "note": "frozen before any evaluation on manifests/final.json",
    }
    json.dump(freeze, open(fz, "w"), indent=1)
    _set("freeze", "completed", methods=list(methods))


def stage_posttrain(seeds=(0,)):
    """RL and its controls from the dev-selected M1 checkpoint (D-014)."""
    for s in seeds:
        init = RUNS / f"m1_pilot_s{s}" / "ckpt_best.pt"
        if not init.exists():
            raise RuntimeError(f"missing {init}")
        for cfg_path, base in (("configs/rl/rl_pilot.yaml", "rl_pilot"), ("configs/rl/filtered_sft_pilot.yaml", "filtered_sft_pilot")):
            name = f"{base}_s{s}"
            if _completed(name):
                continue
            cfg = yaml.safe_load(open(REPO_ROOT / cfg_path))
            cfg["init_from"] = str(init)
            cfg["run_name"] = name
            cfg["seed"] = s
            tmp = RUNS / f"{name}.yaml"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            yaml.safe_dump(cfg, open(tmp, "w"))
            run([PY, "-m", "waymo_av.rl", "--config", str(tmp), "--resume"])
        name = f"extra_sft_pilot_s{s}"
        if not _completed(name):
            _train("configs/train/extra_sft_pilot.yaml", run_name=name, overrides=[f"init_from={init}", f"seed={s}", f"run_name={name}"])
    # matched dev evaluation of the post-trained models into the same eval_dev directory (same scenes/conditions/sampling)
    out = RUNS / "eval_dev"
    ms = []
    for s in seeds:
        for base in ("rl_pilot", "filtered_sft_pilot", "extra_sft_pilot"):
            for fn in ("ckpt_best.pt", "ckpt_final.pt"):
                ck = RUNS / f"{base}_s{s}" / fn
                if ck.exists() and not (out / f"{base}_s{s}" / "clean" / "per_scene.npz").exists():
                    ms.append(f"ckpt:{ck}@{base}_s{s}")
                    break
    if ms:
        run([PY, "-m", "waymo_av.evaluate", "--methods", *ms, "--conditions", *CORE, *HELDOUT, "--manifest", "manifests/dev.json", "--out", str(out), "--tag", "dev"])
    seeds_csv = ",".join(str(s) for s in seeds)
    run([PY, "-m", "waymo_av.aggregate", "--eval-dir", str(out), "--reference", f"m0_pilot_s{seeds[0]}", "--out", "reports/tables/dev"])
    run([PY, "-m", "waymo_av.compare", "--eval-dir", str(out), "--seeds", seeds_csv, "--out", "reports/tables/dev/comparison.md"])
    run([PY, "-m", "waymo_av.figures", "--eval-dir", str(out), "--seeds", seeds_csv, "--out", "reports/figures/dev", "--label", "development set (checkpoint selection; not held-out)"])
    _set("posttrain", "completed", seeds=list(seeds))


def stage_final_eval():
    fz = REPO_ROOT / "manifests" / "freeze.json"
    if not fz.exists():
        raise RuntimeError("final_eval requires manifests/freeze.json (run the freeze stage first)")
    f = json.load(open(fz))
    assert json.load(open(REPO_ROOT / "manifests" / "final.json"))["sha256_ids"] == f["final_manifest_sha256"], "final manifest changed after freeze"
    out = RUNS / "eval_final"
    if (out / "summary_all.json").exists():
        _set("final_eval", "completed", skipped=True)
    else:
        methods = ["cp", "cv", "oracle_log"] + [f"{p}@{n}" if ":" in p else f"ckpt:{p}@{n}" for n, p in f["methods"].items()]
        sm = f["sampling"]
        run([PY, "-m", "waymo_av.evaluate", "--methods", *methods, "--conditions", *f["conditions"], "--manifest", "manifests/final.json", "--out", str(out), "--n-samples", str(sm["n_samples"]), "--audit-samples", str(sm["audit_samples"]), "--temperature", str(sm["temperature"]), "--top-p", str(sm["top_p"]), "--tag", "final"])
        _set("final_eval", "completed", out=str(out))
    ref = next((n for n in f["methods"] if n.startswith("m0_pilot")), None)
    seeds_csv = ",".join(sorted({n.split("_s")[-1] for n in f["methods"] if "_s" in n}))
    run([PY, "-m", "waymo_av.aggregate", "--eval-dir", str(out), "--reference", ref, "--out", "reports/tables/final"])
    run([PY, "-m", "waymo_av.compare", "--eval-dir", str(out), "--seeds", seeds_csv, "--out", "reports/tables/final/comparison.md"])
    run([PY, "-m", "waymo_av.figures", "--eval-dir", str(out), "--seeds", seeds_csv, "--out", "reports/figures/final", "--label", "FINAL held-out set (locked; one pass per frozen method)"])


def stage_report():
    run([PY, "-m", "waymo_av.report", "--dev-tables", "reports/tables/dev", "--final-tables", "reports/tables/final", "--figures", *[str(p) for p in sorted((REPO_ROOT / "reports" / "figures").glob("*.png"))], "--anims", *[str(p) for p in sorted((REPO_ROOT / "reports" / "demos").rglob("*.gif"))]])
    _set("report", "completed")


ORDER = ["audit", "download", "preprocess", "manifests", "calibrate", "overfit", "pilot", "dev_eval", "posttrain", "freeze", "final_eval", "report"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--n-train-shards", type=int, default=24)
    ap.add_argument("--n-val-shards", type=int, default=8)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--until", default=None, help="with --stage all: stop after this stage")
    a = ap.parse_args(argv)
    seeds = tuple(int(s) for s in a.seeds.split(","))
    stages = ORDER if a.stage == "all" else [a.stage]
    if a.until and a.stage == "all":
        stages = ORDER[: ORDER.index(a.until) + 1]
    for st in stages:
        try:
            if st == "audit":
                stage_audit()
            elif st == "download":
                if not stage_download(a.n_train_shards, a.n_val_shards):
                    print("[campaign] BLOCKED at download (dataset access). Stopping.")
                    sys.exit(2)
            elif st == "preprocess":
                stage_preprocess(a.n_train_shards, a.n_val_shards)
            elif st == "manifests":
                stage_manifests()
            elif st == "calibrate":
                stage_calibrate()
            elif st == "overfit":
                stage_overfit()
            elif st == "pilot":
                stage_pilot(seeds)
            elif st == "dev_eval":
                stage_dev_eval(seeds)
            elif st == "posttrain":
                stage_posttrain(seeds)
            elif st == "freeze":
                stage_freeze(seeds)
            elif st == "final_eval":
                stage_final_eval()
            elif st == "report":
                stage_report()
            else:
                raise ValueError(st)
        except SystemExit:
            raise
        except Exception as e:  # noqa
            _set(st, "failed", error=repr(e)[:300])
            raise


if __name__ == "__main__":
    main()
