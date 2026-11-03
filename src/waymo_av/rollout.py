"""Shared inference path: autoregressive paired sampling. Used by evaluation, RL post-training,
demonstrations and animations. Only observed state (`batch` features + d0) enters this function;
ground-truth futures are never read here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from waymo_av.geometry import from_ego_torch
from waymo_av.tokenizer import VerletTokenizer


def _top_p_filter(logits, top_p):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(-1)
    remove = cum - probs > top_p  # keep tokens until cumulative mass exceeds top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, idx, sorted_logits)


@torch.no_grad()
def sample_rollouts(model, batch, n_samples: int, temperature: float = 1.0, top_p: float | None = None, tokenizer: VerletTokenizer | None = None, greedy: bool = False, return_logits: bool = False):
    """Returns dict with
      tokens      [B,S,A,T] int64
      xy_ego      [B,S,A,T,2]   positions in each agent's ego frame
      xy_scene    [B,S,A,T,2]   positions in the scene frame (window origin)
      logp_policy [B,S,A,T]     log-prob of the sampled token under the *unmodified* policy
      logp_sampler[B,S,A,T]     log-prob under the actual sampling distribution (temperature/top-p)
    Sampling with temperature=1 and top_p=None is exact on-policy sampling (logp_policy == logp_sampler)."""
    tokenizer = tokenizer or VerletTokenizer()
    model.eval()
    B = batch["agent_feat"].shape[0]
    A, T = model.cfg.n_agents, model.cfg.n_future
    S = n_samples
    dev = batch["agent_feat"].device
    lat = model.encode(batch)  # [B*A, L, d]
    L, d = lat.shape[1:]
    lat = lat.reshape(B, A, L, d)[:, None].expand(B, S, A, L, d).reshape(B * S * A, L, d)
    tokens = torch.zeros(B * S, A, T, dtype=torch.long, device=dev)
    logp_pol = torch.zeros(B * S, A, T, device=dev)
    logp_smp = torch.zeros(B * S, A, T, device=dev)
    all_logits = [] if return_logits else None
    for t in range(T):
        inp = model.shift_tokens(tokens)[:, :, : t + 1]
        logits = model.decoder(inp, lat)[:, :, t]  # [B*S, A, V]
        logits = logits.float()
        lp_full = F.log_softmax(logits, dim=-1)
        if greedy:
            nxt = logits.argmax(-1)
            lp_s = lp_full
        else:
            l2 = _top_p_filter(logits / temperature, top_p)
            lp_s = F.log_softmax(l2, dim=-1)
            nxt = torch.multinomial(lp_s.exp().reshape(-1, logits.shape[-1]), 1).reshape(B * S, A)
        tokens[:, :, t] = nxt
        logp_pol[:, :, t] = lp_full.gather(-1, nxt[..., None])[..., 0]
        logp_smp[:, :, t] = lp_s.gather(-1, nxt[..., None])[..., 0]
        if return_logits:
            all_logits.append(logits)
    # decode positions
    act = tokenizer.torch_action_table(dev).float()
    d0 = batch["d0"][:, None].expand(B, S, A, 2).reshape(B * S, A, 2).float()
    p = torch.zeros(B * S, A, 2, device=dev)
    dlt = d0.clone()
    xy_ego = torch.zeros(B * S, A, T, 2, device=dev)
    for t in range(T):
        p, dlt = tokenizer.torch_step(p, dlt, act[tokens[:, :, t]])
        xy_ego[:, :, t] = p
    origin = batch["ego_origin"][:, None].expand(B, S, A, 2).reshape(B * S * A, 2).float()
    heading = batch["ego_heading"][:, None].expand(B, S, A).reshape(B * S * A).float()
    xy_scene = from_ego_torch(xy_ego.reshape(B * S * A, T, 2), origin, heading).reshape(B, S, A, T, 2)
    out = dict(
        tokens=tokens.reshape(B, S, A, T),
        xy_ego=xy_ego.reshape(B, S, A, T, 2),
        xy_scene=xy_scene,
        logp_policy=logp_pol.reshape(B, S, A, T),
        logp_sampler=logp_smp.reshape(B, S, A, T),
    )
    if return_logits:
        out["logits"] = torch.stack(all_logits, dim=2).reshape(B, S, A, T, -1)
    return out


def decode_tokens_to_scene(tokens, batch, tokenizer: VerletTokenizer | None = None):
    """tokens [B,A,T] (e.g. ground-truth tokens) -> scene-frame positions [B,A,T,2]."""
    tokenizer = tokenizer or VerletTokenizer()
    B, A, T = tokens.shape
    dev = tokens.device
    act = tokenizer.torch_action_table(dev).float()
    p = torch.zeros(B, A, 2, device=dev)
    dlt = batch["d0"].float().clone()
    xy = torch.zeros(B, A, T, 2, device=dev)
    for t in range(T):
        p, dlt = tokenizer.torch_step(p, dlt, act[tokens[:, :, t]])
        xy[:, :, t] = p
    return from_ego_torch(xy.reshape(B * A, T, 2), batch["ego_origin"].reshape(B * A, 2).float(), batch["ego_heading"].reshape(B * A).float()).reshape(B, A, T, 2)
