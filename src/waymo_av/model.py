"""Compact MotionLM-style model: early-fusion latent-query scene encoder (per target ego frame)
and a causal transformer decoder over the flattened (agent, time) motion-token sequence.

Causality (MotionLM Sec. 3.2): the decoder input at (agent a, step t) is the token a_{t-1} (a learned
START token at t=0). Position (a,t) may attend to (a',t') iff t' <= t. Hence when predicting a_t it
sees every agent's tokens up to a_{t-1} and never a_t itself or any other agent's step-t token.
The `independent` variant additionally requires a' == a (marginal ablation).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from waymo_av.features import AGENT_FEAT_DIM, MAP_FEAT_DIM, TL_FEAT_DIM


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_latents: int = 64
    enc_layers: int = 2
    dec_layers: int = 4
    vocab_size: int = 169
    n_future: int = 16
    n_agents: int = 2
    n_hist: int = 11
    max_agent_slots: int = 32
    dropout: float = 0.1
    independent: bool = False  # marginal ablation: no cross-agent attention in the decoder

    def to_dict(self):
        return asdict(self)


class MHA(nn.Module):
    def __init__(self, d, h, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)

    def forward(self, q, kv, key_padding_mask=None, attn_mask=None):
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        return out


class Block(nn.Module):
    """Pre-norm transformer block with optional cross-attention."""

    def __init__(self, d, h, d_ff, dropout, cross=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = MHA(d, h, dropout)
        self.cross = cross
        if cross:
            self.lnc = nn.LayerNorm(d)
            self.cross_attn = MHA(d, h, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, self_mask=None, self_kpm=None, ctx=None, ctx_kpm=None):
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, key_padding_mask=self_kpm, attn_mask=self_mask))
        if self.cross:
            x = x + self.drop(self.cross_attn(self.lnc(x), ctx, key_padding_mask=ctx_kpm))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class SceneEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.agent_proj = nn.Linear(AGENT_FEAT_DIM, d)
        self.map_proj = nn.Linear(MAP_FEAT_DIM, d)
        self.tl_proj = nn.Linear(TL_FEAT_DIM, d)
        self.modality_emb = nn.Embedding(3, d)
        self.missing_emb = nn.Parameter(torch.zeros(d))  # learned token for unobserved agent steps (D-011)
        self.slot_emb = nn.Embedding(cfg.max_agent_slots, d)
        self.time_emb = nn.Embedding(cfg.n_hist, d)
        self.latents = nn.Parameter(torch.randn(cfg.n_latents, d) * 0.02)
        self.ln_in = nn.LayerNorm(d)
        self.cross = Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout, cross=True)
        self.layers = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.enc_layers)])
        self.ln_out = nn.LayerNorm(d)
        nn.init.normal_(self.missing_emb, std=0.02)

    def forward(self, agent_feat, agent_valid, map_feat, map_valid, tl_feat, tl_valid):
        """All inputs with a leading batch of B*2 (ego frames folded into batch).
        agent_feat [N, S, H, F]; agent_valid [N, S, H]; map_feat [N, M, F]; tl_feat [N, L, F]."""
        N, S, Hh, _ = agent_feat.shape
        d = self.cfg.d_model
        a = self.agent_proj(agent_feat)
        slot = self.slot_emb(torch.arange(S, device=a.device))[None, :, None, :]
        tim = self.time_emb(torch.arange(Hh, device=a.device))[None, None, :, :]
        # unobserved steps: features were zeroed upstream; replace with the missing embedding so the
        # model can still see *which* slots/steps are missing (never physical zeros as positions).
        a = torch.where(agent_valid[..., None], a, self.missing_emb.expand_as(a))
        a = a + slot + tim + self.modality_emb.weight[0]
        # keys: all agent steps for agents that have any observation (missing steps kept as tokens),
        # fully-absent slots are masked out.
        slot_present = agent_valid.any(-1)  # [N,S]
        a_kpm = ~slot_present[:, :, None].expand(N, S, Hh)
        a = a.reshape(N, S * Hh, d)
        a_kpm = a_kpm.reshape(N, S * Hh)
        m = self.map_proj(map_feat) + self.modality_emb.weight[1]
        t = self.tl_proj(tl_feat) + self.modality_emb.weight[2]
        tokens = torch.cat([a, m, t], dim=1)
        kpm = torch.cat([a_kpm, ~map_valid, ~tl_valid], dim=1)
        # guard: a row with all keys masked would produce NaNs; the own-agent current step is always present
        tokens = self.ln_in(tokens)
        lat = self.latents[None].expand(N, -1, -1)
        x = lat + self.cross.cross_attn(self.cross.lnc(lat), tokens, key_padding_mask=kpm)
        x = x + self.cross.ff(self.cross.ln2(x))
        for layer in self.layers:
            x = layer(x)
        return self.ln_out(x)  # [N, n_latents, d]


def build_decoder_mask(n_agents: int, n_steps: int, independent: bool, device=None) -> torch.Tensor:
    """Boolean attn_mask [A*T, A*T] with True = NOT allowed (PyTorch convention). Ordering is
    time-major: index = t * A + a. Allowed iff t' <= t (and a' == a if independent)."""
    t = torch.arange(n_steps, device=device).repeat_interleave(n_agents)
    a = torch.arange(n_agents, device=device).repeat(n_steps)
    allowed = t[None, :] <= t[:, None]
    if independent:
        allowed = allowed & (a[None, :] == a[:, None])
    return ~allowed


class MotionDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size + 1, d)  # last index = START
        self.start_token = cfg.vocab_size
        self.time_emb = nn.Embedding(cfg.n_future, d)
        self.agent_emb = nn.Embedding(cfg.n_agents, d)
        self.layers = nn.ModuleList([Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout, cross=True) for _ in range(cfg.dec_layers)])
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size)

    def forward(self, input_tokens, scene_latents):
        """input_tokens [B, A, T] (shifted: START then a_0..a_{T-2}); scene_latents [B*A, L, d]
        (latents of agent a's own ego frame at index b*A + a). Returns logits [B, A, T, V]."""
        B, A, T = input_tokens.shape
        d = self.cfg.d_model
        x = self.tok_emb(input_tokens)
        x = x + self.time_emb(torch.arange(T, device=x.device))[None, None]
        x = x + self.agent_emb(torch.arange(A, device=x.device))[None, :, None]
        mask = build_decoder_mask(A, T, self.cfg.independent, device=x.device)
        # time-major flattening so that the mask indexing matches: [B, T, A, d] -> [B, T*A, d]
        x = x.permute(0, 2, 1, 3).reshape(B, T * A, d)
        for layer in self.layers:
            h = layer.ln1(x)
            x = x + layer.drop(layer.self_attn(h, h, attn_mask=mask))
            # cross-attention per agent to its own scene latents
            xa = x.reshape(B, T, A, d).permute(0, 2, 1, 3).reshape(B * A, T, d)
            xa = xa + layer.drop(layer.cross_attn(layer.lnc(xa), scene_latents))
            x = xa.reshape(B, A, T, d).permute(0, 2, 1, 3).reshape(B, T * A, d)
            x = x + layer.drop(layer.ff(layer.ln2(x)))
        x = self.ln_out(x).reshape(B, T, A, d).permute(0, 2, 1, 3)
        return self.head(x)


class MotionLMStyle(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SceneEncoder(cfg)
        self.decoder = MotionDecoder(cfg)

    def encode(self, batch):
        B = batch["agent_feat"].shape[0]
        A = self.cfg.n_agents
        fold = lambda x: x.reshape(B * A, *x.shape[2:])
        return self.encoder(
            fold(batch["agent_feat"]), fold(batch["agent_valid"]), fold(batch["map_feat"]), fold(batch["map_valid"]), fold(batch["tl_feat"]), fold(batch["tl_valid"])
        )

    def shift_tokens(self, tokens):
        """tokens [B,A,T] -> decoder inputs [B,A,T]: START, a_0, ..., a_{T-2}."""
        start = torch.full_like(tokens[..., :1], self.decoder.start_token)
        return torch.cat([start, tokens[..., :-1]], dim=-1)

    def forward(self, batch, tokens=None):
        """Teacher-forced logits [B,A,T,V] for the given target tokens (defaults to batch['tokens'])."""
        tokens = batch["tokens"] if tokens is None else tokens
        lat = self.encode(batch)
        return self.decoder(self.shift_tokens(tokens), lat)

    def nll(self, batch, reduction="mean"):
        logits = self.forward(batch)
        tgt = batch["tokens"]
        valid = batch["token_valid"]
        lp = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        lp = lp * valid
        if reduction == "mean":
            return lp.sum() / valid.sum().clamp(min=1)
        return lp

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
