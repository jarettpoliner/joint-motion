"""Initial-motion follow-up (FU-IM; exploratory on dev, one confirmation pass). Inference only.

Question: how much of the frozen models' noise sensitivity enters through the initial-motion
estimate d0, and does corruption training (M1) still help once a stronger d0 estimator is used?

Mechanism (traced in reports/followup_initial_motion/PROTOCOL.md):
  * d0 = tokenizer.initial_delta_from_observation(observed target history) — least squares over
    the valid frames among history indices 5..10 (0.5 s, nominal 0.1 s spacing), fallbacks for
    < 2 valid frames. It is rotated into each target's ego frame (features.build_features).
  * d0 is used (i) as the Verlet start state when integrating sampled tokens (rollout.py),
    (ii) to tokenise TRAINING labels relative to d0 (features.labels_from_window), and (iii) by
    the CV baseline. It is NOT an input of the network (model.encode reads agent/map/TL features
    only). Hence replacing d0 at decode time changes neither the network inputs nor the sampled
    token sequences: arms below share identical tokens and differ only in integration.

Arms (all decode the SAME sampled tokens):
  orig       d0 from the version-1 estimator on the model-visible (possibly corrupted) history
  oracle     ORACLE DIAGNOSTIC: the version-1 estimator on the UNCORRUPTED observed history
             (history indices 0..10 only; never future positions/validity). Unavailable in deployment.
  ols1s      candidate 1: OLS over all valid frames of the full 1.0 s history, actual timestamps
  anchored1s candidate 2: LS line constrained through the observed current anchor over the full
             1.0 s history, actual timestamps
  xd0:<cond> (diagnostic, clean-condition pass only) clean-input tokens decoded with the d0 the
             original estimator produces under corruption <cond>: the d0 path alone, features clean.

Version-1 code paths are not modified; this module only adds functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from waymo_av.corruption import CUR, H, Observation, apply_corruption, deterministic_seed, spec_from_name
from waymo_av.dataset import WindowDataset, collate, load_window, read_manifest
from waymo_av.geometry import from_ego_torch, to_ego
from waymo_av.paths import REPO_ROOT, load_paths
from waymo_av.tokenizer import N_FUTURE, STEP_DT, VerletTokenizer, initial_delta_from_observation

CONDITIONS = ["clean", "noise_0.1", "noise_0.3", "noise_0.5", "noise_1", "bias_0.5"]
CORRUPTED = CONDITIONS[1:]
CANDIDATES = ["ols1s", "anchored1s"]
OUT_ROOT_NAME = "followup_initmotion"
TRACKED = REPO_ROOT / "reports" / "followup_initial_motion"
SAMPLING = {"n_samples": 6, "temperature": 1.0, "top_p": 0.95}  # = version-1 frozen sampling (no 32-sample audit)
CORRUPTION_GLOBAL_SEED = 0  # = version-1 evaluate.py --seed 0: identical corruption realisations


# ----------------------------------------------------------------------------- estimators
# All estimators: xy [11,2] scene-frame history, valid [11] bool, ts [11] seconds (actual
# timestamps). Return the initial delta in METRES PER 0.5 s STEP (scene frame), i.e. velocity
# [m/s] * STEP_DT, the unit the Verlet tokenizer integrates. Invalid frames are never read.

def est_orig(xy, valid, ts=None):
    """Version-1 estimator, called unchanged (tokenizer.initial_delta_from_observation): OLS over
    the valid frames among history indices 5..10 (0.5 s) at NOMINAL 0.1 s spacing; if fewer than
    two are valid, two-point difference to the most recent earlier valid frame (up to 1 s back);
    zero if the current frame is invalid or no earlier valid frame exists. `ts` is ignored."""
    return initial_delta_from_observation(np.asarray(xy, np.float64), np.asarray(valid, bool))


def _rel_t(ts):
    ts = np.asarray(ts, np.float64)
    return ts - ts[CUR]


def est_ols1s(xy, valid, ts):
    """OLS velocity over ALL valid frames of the full 1.0 s history (indices 0..10) using actual
    timestamps. Zero if the current frame is invalid or fewer than two frames are valid."""
    xy = np.asarray(xy, np.float64)
    valid = np.asarray(valid, bool)
    if not valid[CUR]:
        return np.zeros(2)
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.zeros(2)
    t = _rel_t(ts)[idx]
    tc = t - t.mean()
    v = (tc[:, None] * (xy[idx] - xy[idx].mean(0))).sum(0) / (tc**2).sum()
    return v * STEP_DT


def est_anchored1s(xy, valid, ts):
    """Least-squares velocity of a line constrained to pass through the observed current anchor
    (index 10, the position every forecast starts from), fitted to all valid earlier frames of the
    1.0 s history with actual timestamps: v = sum t (x_t - x_0) / sum t^2. Zero if the current
    frame is invalid or no earlier frame is valid."""
    xy = np.asarray(xy, np.float64)
    valid = np.asarray(valid, bool)
    if not valid[CUR]:
        return np.zeros(2)
    idx = np.flatnonzero(valid[:CUR])
    if len(idx) == 0:
        return np.zeros(2)
    t = _rel_t(ts)[idx]
    v = (t[:, None] * (xy[idx] - xy[CUR])).sum(0) / (t**2).sum()
    return v * STEP_DT


ESTIMATORS = {"orig": est_orig, "ols1s": est_ols1s, "anchored1s": est_anchored1s}


def ego_d0(est, xy2, valid2, ts, heading2) -> np.ndarray:
    """[2,2] ego-frame d0 for both targets (same rotation as features.build_features)."""
    out = np.zeros((2, 2), dtype=np.float32)
    for i in range(2):
        d = est(xy2[i], valid2[i], ts)
        out[i] = to_ego(np.asarray(d, np.float64)[None], np.zeros(2), float(heading2[i]))[0]
    return out


# ----------------------------------------------------------------------------- dataset
class InitMotionDataset(WindowDataset):
    """Version-1 WindowDataset item (identical corrupted features, labels and d0) plus history-only
    d0 variants. Keys added:
      d0_<est>         estimator <est> on the model-visible (corrupted) target history
      d0clean_<est>    estimator <est> on the uncorrupted observed history (oracle reference)
      xd0_<cond>       version-1 estimator under corruption <cond> (only if cross_conditions)
      evalonly_first_disp  [2,2] ego-frame logged displacement over the first 0.5 s and its validity
                           (FUTURE-DERIVED; used only to score estimates, never passed to decoding)"""

    def __init__(self, files, corruption="clean", cross_conditions=(), **kw):
        super().__init__(files, corruption=corruption, global_seed=CORRUPTION_GLOBAL_SEED, return_window=True, **kw)
        self.cross_conditions = list(cross_conditions)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        win = load_window(self.files[idx])
        ts = win.timestamps[:H].astype(np.float64)  # history timestamps only
        clean = Observation.from_window(win)  # uncorrupted OBSERVED history (indices 0..10)
        ox, ov = item["obs_target_xy"].numpy(), item["obs_target_valid"].numpy()
        hd = item["ego_heading"].numpy()
        for name, fn in ESTIMATORS.items():
            item[f"d0_{name}"] = torch.as_tensor(ego_d0(fn, ox, ov, ts, hd))
            item[f"d0clean_{name}"] = torch.as_tensor(ego_d0(fn, clean.xy[0:2], clean.valid[0:2], ts, hd))
        if not torch.equal(item["d0_orig"], item["d0"]):
            raise AssertionError(f"{win.scenario_id}: recomputed version-1 d0 differs from features d0")
        item["d0_oracle"] = item["d0clean_orig"].clone()
        for c in self.cross_conditions:
            spec = spec_from_name(c)
            rng = np.random.default_rng(deterministic_seed(win.scenario_id, spec.name, CORRUPTION_GLOBAL_SEED))
            o = apply_corruption(clean, spec, rng) if spec.family != "clean" else clean
            item[f"xd0_{spec.name}"] = torch.as_tensor(ego_d0(est_orig, o.xy[0:2], o.valid[0:2], ts, hd))
        # evaluation-only reference (future-derived): never enters decoding
        fd = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            fd[i] = to_ego((item["gt_future_xy"][i, 0].numpy() - item["cur_xy"][i].numpy()).astype(np.float64)[None], np.zeros(2), float(hd[i]))[0]
        item["evalonly_first_disp"] = torch.as_tensor(fd)
        item["evalonly_first_valid"] = item["gt_future_valid"][:, 0].clone()
        return item


# ----------------------------------------------------------------------------- decoding
def decode_with_d0(tokens, d0, origin, heading, tokenizer=None):
    """tokens [B,S,A,T] long; d0 [B,A,2] (ego frame, m per 0.5 s); origin [B,A,2]; heading [B,A]
    -> scene-frame positions [B,S,A,T,2]. Same float32 operations, in the same order, as
    rollout.sample_rollouts, so decode_with_d0(tokens, batch['d0'], ...) reproduces its xy_scene."""
    tokenizer = tokenizer or VerletTokenizer()
    B, S, A, T = tokens.shape
    dev = tokens.device
    act = tokenizer.torch_action_table(dev).float()
    tk = tokens.reshape(B * S, A, T)
    p = torch.zeros(B * S, A, 2, device=dev)
    dlt = d0[:, None].expand(B, S, A, 2).reshape(B * S, A, 2).float().clone()
    xy = torch.zeros(B * S, A, T, 2, device=dev)
    for t in range(T):
        p, dlt = tokenizer.torch_step(p, dlt, act[tk[:, :, t]])
        xy[:, :, t] = p
    o = origin[:, None].expand(B, S, A, 2).reshape(B * S * A, 2).float()
    h = heading[:, None].expand(B, S, A).reshape(B * S * A).float()
    return from_ego_torch(xy.reshape(B * S * A, T, 2), o, h).reshape(B, S, A, T, 2)


def zero_tokens(B, A, tokenizer=None, device="cpu"):
    tokenizer = tokenizer or VerletTokenizer()
    return torch.full((B, 1, A, N_FUTURE), tokenizer.cfg.zero_token, dtype=torch.long, device=device)


# ----------------------------------------------------------------------------- metrics
def per_step_joint_min_error(pred, gt, valid):
    """pred [B,S,A,T,2], gt [B,A,T,2], valid [B,A,T] -> [B,T]: at each 2 Hz step k, min over joint
    samples of the mean error over both targets (NaN unless both targets are valid at k)."""
    err = np.linalg.norm(pred - gt[:, None], axis=-1)  # [B,S,A,T]
    both = valid.all(1)  # [B,T]
    m = err.mean(2).min(1)
    return np.where(both, m, np.nan)


def metrics_job(pred, g, offroad_sign):
    from waymo_av.evaluate import scene_metrics_g

    out = scene_metrics_g(pred, g, offroad_sign, False)
    out["step_jmin_err"] = per_step_joint_min_error(pred, g["gt"], g["v"])
    return out


# ----------------------------------------------------------------------------- provenance
def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def sampling_seed(model_tag: str, cond: str) -> int:
    """Documented torch seed per (model, condition); arms share the draw."""
    return zlib.crc32(f"fu_im|{model_tag}|{cond}".encode())


def out_root() -> Path:
    return Path(load_paths()["runs_dir"]) / OUT_ROOT_NAME


class Ledger:
    """Follow-up GPU ledger (separate from the version-1 registry/gpu_hours.jsonl)."""

    def __init__(self, path: Path, cap_hours: float):
        self.path, self.cap = path, cap_hours
        path.parent.mkdir(parents=True, exist_ok=True)

    def used_hours(self):
        if not self.path.exists():
            return 0.0
        return sum(json.loads(l)["gpu_seconds"] for l in open(self.path) if l.strip()) / 3600.0

    def log(self, job, seconds, note=""):
        with open(self.path, "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "job": job, "gpu_seconds": seconds, "note": note}) + "\n")


# ----------------------------------------------------------------------------- runner
def _job_want(stage, model_tag, ckpt, cond, files, arms, cross, manifest_path):
    return {"stage": stage, "model_tag": model_tag, "ckpt": str(ckpt) if ckpt else None, "condition": cond, "arms": list(arms), "cross": list(cross),
            "n_scenes": len(files), "sampling": SAMPLING if ckpt else None, "sampling_seed": sampling_seed(model_tag, cond) if ckpt else None,
            "corruption_global_seed": CORRUPTION_GLOBAL_SEED, "manifest": str(manifest_path) if manifest_path else None,
            "scenario_ids_sha256": hashlib.sha256("\n".join(Path(f).stem for f in files).encode()).hexdigest()}


def _check_existing(p: Path, want: dict):
    if not p.exists():
        return None
    have = json.load(open(p))
    if all(have.get(k) == v for k, v in want.items()):
        return have
    raise RuntimeError(f"{p} exists with different provenance; refusing to overwrite")


def sample_job(stage, model_tag, ckpt, cond, files, arms, out_dir, *, cross=(), batch_size=32, device="cuda", ledger=None, manifest_path=None):
    """Phase 1 (GPU). Sample the model's joint tokens ONCE per scene and decode them under every
    arm's d0. Writes <out_dir>/<model_tag>/<cond>/{preds.npz, diag.npz, geom.npz, sample_provenance.json}
    atomically; an existing job is reused only if its provenance matches exactly."""
    from waymo_av.evaluate import _to_dev
    from waymo_av.rollout import sample_rollouts
    from waymo_av.train import load_model_from_ckpt

    cond = spec_from_name(cond).name if cond != "clean" else cond
    jdir = out_dir / model_tag / cond
    want = _job_want(stage, model_tag, ckpt, cond, files, arms, cross, manifest_path)
    have = _check_existing(jdir / "sample_provenance.json", want)
    if have:
        print(f"[{stage}] {model_tag}/{cond}: sampled with matching provenance — reused", flush=True)
        return have
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = None
    if ckpt:
        model, _ = load_model_from_ckpt(ckpt, dev)
        model.eval()
    tok = VerletTokenizer()
    ds = InitMotionDataset(files, corruption=cond, cross_conditions=cross)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)
    if model is not None:
        torch.manual_seed(want["sampling_seed"])
    arm_list = list(arms) + [f"xd0:{spec_from_name(c).name}" for c in cross]
    preds = {a: [] for a in arm_list}
    diag, geom, ids = {}, {}, []
    t_start = time.time()
    n_equal = 0
    for batch in dl:
        b = _to_dev(batch, dev)
        ids.extend(batch["scenario_id"])
        B = len(batch["scenario_id"])
        if model is not None:
            r = sample_rollouts(model, b, n_samples=SAMPLING["n_samples"], temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"], tokenizer=tok)
            tokens = r["tokens"]
        else:  # deterministic CV reference: zero-action tokens = constant-velocity continuation of d0
            r = None
            tokens = zero_tokens(B, 2, tok, dev)
        for a in arm_list:
            key = f"xd0_{a.split(':', 1)[1]}" if a.startswith("xd0:") else f"d0_{a}"
            xy = decode_with_d0(tokens, b[key], b["ego_origin"], b["ego_heading"], tok)
            if a == "orig" and r is not None:
                if not torch.equal(xy, r["xy_scene"]):
                    raise AssertionError("decode_with_d0 does not reproduce sample_rollouts for the original d0")
                n_equal += 1
            preds[a].append(xy.cpu().numpy())
        dd = {k: b[k].cpu().numpy() for k in b if isinstance(k, str) and k.startswith(("d0_", "d0clean_", "xd0_", "evalonly_"))}
        act = tok.torch_action_table(tokens.device).float()[tokens]  # [B,S,A,T,2] ego-frame actions (m per step^2)
        dd["cum_action_mean"] = act.cumsum(3).mean(1)[:, :, :4].cpu().numpy()  # [B,A,4,2] sample-mean cumulative action, steps 1..4
        dd["tokens"] = tokens.cpu().numpy().astype(np.int16)
        for k, v in dd.items():
            diag.setdefault(k, []).append(v)
        for k, src in (("gt", "gt_future_xy"), ("v", "gt_future_valid"), ("cur_xy", "cur_xy"), ("cur_h", "cur_heading"), ("size", "target_size"), ("types", "target_type")):
            v = batch[src].cpu().numpy()
            geom.setdefault(k, []).append(v[..., :2] if k == "size" else v)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t_start
    tmp = jdir.with_name(jdir.name + ".partial")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.savez(tmp / "preds.npz", scenario_ids=np.array(ids), **{a.replace(":", "__"): np.concatenate(v) for a, v in preds.items()})
    np.savez(tmp / "diag.npz", scenario_ids=np.array(ids), **{k: np.concatenate(v) for k, v in diag.items()})
    np.savez(tmp / "geom.npz", scenario_ids=np.array(ids), window_paths=np.array([str(f) for f in files]), **{k: np.concatenate(v) for k, v in geom.items()})
    prov = dict(want)
    prov.update({"git_sha": git_sha(), "ckpt_sha256": sha256_file(ckpt) if ckpt else None, "decode_equality_checks_passed": n_equal,
                 "n_batches": len(diag["tokens"]), "wall_s": wall, "device": str(dev), "torch": torch.__version__, "t_done": time.strftime("%Y-%m-%dT%H:%M:%S")})
    json.dump(prov, open(tmp / "sample_provenance.json", "w"), indent=1)
    if jdir.exists():
        import shutil

        shutil.rmtree(jdir)  # only reachable when no provenance file existed (incomplete earlier job)
    tmp.rename(jdir)
    if ledger is not None:
        ledger.log(f"{stage}/{model_tag}/{cond}", wall, note="phase-1 process wall-clock while holding the GPU (feature building + sampling + decoding)")
    print(f"[{stage}] sampled {model_tag}/{cond} n={len(ids)} in {wall:.0f}s", flush=True)
    return prov


def metrics_for_job(jdir: Path, offroad_sign=1, workers=8, chunk=32):
    """Phase 2 (CPU only). Version-1 scene metrics (evaluate.scene_metrics_g) for every arm in
    preds.npz, plus per-step joint error curves. Writes <jdir>/metrics/<arm>.npz atomically."""
    from concurrent.futures import ProcessPoolExecutor

    from waymo_av.metrics.physical import EdgeTree

    mdir = jdir / "metrics"
    if (mdir / "done.json").exists():
        return json.load(open(mdir / "done.json"))
    P = np.load(jdir / "preds.npz")
    G = np.load(jdir / "geom.npz")
    ids = P["scenario_ids"]
    assert np.array_equal(ids, G["scenario_ids"])
    wins = [load_window(Path(p)) for p in G["window_paths"]]
    assert [w.scenario_id for w in wins] == list(map(str, ids)), "window order differs from sampled order"
    trees = [EdgeTree(w.edge_xy, w.edge_poly) for w in wins]
    N = len(ids)
    gs = []
    for s in range(0, N, chunk):
        sl = slice(s, s + chunk)
        gs.append({"gt": G["gt"][sl], "v": G["v"][sl], "cur_xy": G["cur_xy"][sl], "cur_h": G["cur_h"][sl], "size": G["size"][sl], "types": G["types"][sl], "trees": trees[sl]})
    arms = [k for k in P.files if k != "scenario_ids"]
    t0 = time.time()
    tmp = jdir / "metrics.partial"
    tmp.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {a: [ex.submit(metrics_job, P[a][s : s + chunk], gs[i], offroad_sign) for i, s in enumerate(range(0, N, chunk))] for a in arms}
        summ = {}
        for a in arms:
            per = {}
            for f in futs[a]:
                for k, v in f.result().items():
                    per.setdefault(k, []).append(np.asarray(v, dtype=np.float64))
            arr = {k: np.concatenate(v) for k, v in per.items()}
            np.savez(tmp / f"{a}.npz", scenario_ids=ids, **arr)
            summ[a] = {k: float(np.nanmean(arr[k])) for k in ("joint_minADE_8s", "joint_minFDE_8s", "overlap_rate_jointvalid", "offroad_frac")}
    done = {"arms": arms, "n_scenes": int(N), "cpu_wall_s": time.time() - t0, "workers": workers, "offroad_sign": offroad_sign, "summary": summ, "t_done": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(done, open(tmp / "done.json", "w"), indent=1)
    tmp.rename(mdir)
    return done


STAGES = {
    # stage: (manifest, limit, seeds, arms) — arms for the dev/confirm stages are completed from selection.json
    "diag": ("manifests/dev.json", 200, (0,), ["orig", "oracle", "ols1s", "anchored1s"]),
    "dev": ("manifests/dev.json", None, (0, 1, 2), None),
    "confirm": (None, None, (0, 1, 2), None),
}


def stage_arms(stage):
    if STAGES[stage][3] is not None:
        return STAGES[stage][3]
    sel = json.load(open(TRACKED / "selection.json"))
    chosen = sel["selected"] or sel["best_scoring_candidate"]
    return ["orig", chosen, "oracle"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--models", nargs="+", default=["cv", "m0_pilot", "m1_pilot"])
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gpu-cap-hours", type=float, default=3.0)
    ap.add_argument("--sample-only", action="store_true", help="phase 1 only (GPU); metrics later")
    a = ap.parse_args(argv)
    paths = load_paths()
    runs = Path(paths["runs_dir"])
    man_rel, limit, seeds, _ = STAGES[a.stage]
    if a.stage == "confirm":
        frz = TRACKED / "freeze.json"
        if not frz.exists():
            raise SystemExit("confirmation requires reports/followup_initial_motion/freeze.json (estimator and comparisons frozen first)")
        man_rel = json.load(open(frz))["confirm_manifest"]
    man = REPO_ROOT / man_rel
    files = read_manifest(man)
    if limit:
        files = files[:limit]
    arms = stage_arms(a.stage)
    out_dir = out_root() / a.stage
    ledger = Ledger(out_root() / "gpu_ledger.jsonl", a.gpu_cap_hours)
    sign = json.load(open(REPO_ROOT / "manifests" / "metric_config.json"))["offroad_sign"]
    jobs = []
    for m in a.models:
        if m == "cv":
            jobs.append(("cv", None))
            continue
        for s in seeds:
            jobs.append((f"{m}_s{s}", runs / f"{m}_s{s}" / "ckpt_best.pt"))
    done_dirs = []
    for tag, ck in jobs:
        for c in a.conditions:
            used = ledger.used_hours()
            if used >= a.gpu_cap_hours:
                print(f"GPU cap reached ({used:.2f} h >= {a.gpu_cap_hours} h); stopping. Re-run the same command to resume.", flush=True)
                return 2
            cross = CORRUPTED if (a.stage == "diag" and c == "clean" and ck is not None) else ()
            sample_job(a.stage, tag, ck, c, files, arms, out_dir, cross=cross, batch_size=a.batch_size, ledger=ledger, manifest_path=man_rel)
            done_dirs.append(out_dir / tag / (spec_from_name(c).name if c != "clean" else c))
    if a.sample_only:
        print(f"sampling done; follow-up GPU ledger total {ledger.used_hours():.3f} h", flush=True)
        return 0
    torch.cuda.empty_cache()
    for jd in done_dirs:
        d = metrics_for_job(jd, offroad_sign=sign, workers=a.workers)
        print(f"[{a.stage}] metrics {jd.parent.name}/{jd.name}: " + "  ".join(f"{k}:{v['joint_minADE_8s']:.3f}" for k, v in d["summary"].items()), flush=True)
    print(f"done; follow-up GPU ledger total {ledger.used_hours():.3f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
