import math

import torch
import torch.nn.functional as F
from torch import nn

from tilde_winner.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * norm


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, device=device).float() / half_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D/2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    rot_even = (x_even * cos) - (x_odd * sin)
    rot_odd = (x_even * sin) + (x_odd * cos)
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.d_model)
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.rope_theta = cfg.rope_theta
        self.dropout = nn.Dropout(cfg.dropout)

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        need_confidence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = _build_rope_cache(seq_len, self.head_dim, x.device, self.rope_theta)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self.n_heads != self.n_kv_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal, -torch.finfo(scores.dtype).max)
        probs = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(out)

        if not need_confidence:
            return out, None

        probs_f = probs.float()
        entropy = -(probs_f.clamp_min(1e-9) * probs_f.clamp_min(1e-9).log()).sum(dim=-1)
        log_positions = torch.arange(1, seq_len + 1, device=x.device, dtype=probs_f.dtype).log()
        max_entropy = log_positions.clamp_min(1.0).view(1, 1, seq_len)
        entropy_norm = (entropy / max_entropy).clamp(0.0, 1.0)
        confidence = 1.0 - entropy_norm.mean(dim=1)
        return out, confidence


class BaselineBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, _ = self.attn(self.attn_norm(x), need_confidence=False)
        x = x + a
        m = self.ffn(self.ffn_norm(x))
        x = x + m
        zero = x.new_zeros(())
        return x, zero, zero, zero


class EGMBlock(nn.Module):
    """
    Entropy-Gated Mixer (EGM) block:
    - Compute attention and MLP in parallel from the same normalized stream
    - Mix them token-wise with a learned gate conditioned on attention confidence
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.pre_norm = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = SwiGLU(cfg)
        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model + 1, cfg.d_model // 4),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 4, 1),
        )
        self.resid_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.pre_norm(x)
        a, confidence = self.attn(h, need_confidence=True)
        m = self.ffn(h)
        assert confidence is not None

        gate_input = torch.cat((h, confidence.unsqueeze(-1)), dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        mixed = (gate * a) + ((1.0 - gate) * m)
        x = x + (self.resid_scale * mixed)

        aux = F.mse_loss(gate.squeeze(-1), confidence.detach())
        return x, aux, gate.mean().detach(), confidence.mean().detach()


class DecoderLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        block_cls = EGMBlock if cfg.variant == "egm" else BaselineBlock
        self.blocks = nn.ModuleList(block_cls(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, float]]:
        x = self.tok_emb(idx)
        x = self.drop(x)

        aux_losses: list[torch.Tensor] = []
        gate_means: list[torch.Tensor] = []
        conf_means: list[torch.Tensor] = []

        for block in self.blocks:
            x, aux, gate_mean, conf_mean = block(x)
            aux_losses.append(aux)
            gate_means.append(gate_mean)
            conf_means.append(conf_mean)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        aux_loss = torch.stack(aux_losses).mean() if aux_losses else logits.new_zeros(())
        metrics = {
            "gate_mean": float(torch.stack(gate_means).mean().item()) if gate_means else 0.0,
            "confidence_mean": float(torch.stack(conf_means).mean().item()) if conf_means else 0.0,
        }
        return logits, loss, aux_loss, metrics
