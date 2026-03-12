"""
Closed-form Continuous-time (CfC) Liquid cell.

Drop-in replacement for CausalSelfAttention. Instead of building an O(N²)
attention matrix, each token updates a hidden state h via a gated ODE solved
in closed form — O(N) memory, O(N) compute.

CfC closed-form (from Hasani et al., 2022):
    f = σ( W_f · [x; h] )       -- forget/decay gate
    g = tanh( W_g · [x; h] )    -- input candidate
    h' = f * h + (1 - f) * g    -- new hidden state

Reference: https://arxiv.org/abs/2106.13898
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import Linear


class CfCCell(nn.Module):
    """Single time-step CfC update: (x: [B,C], h: [B,C]) -> h_new: [B,C]"""

    def __init__(self, n_embd):
        super().__init__()
        # Both gates map [x; h] (size 2*n_embd) -> n_embd
        self.W_f = Linear(2 * n_embd, n_embd, bias=False)
        self.W_g = Linear(2 * n_embd, n_embd, bias=False)

    def forward(self, x, h):
        xh = torch.cat([x, h], dim=-1)          # (B, 2C)
        f = torch.sigmoid(self.W_f(xh))          # forget gate: how much of h to keep
        g = torch.tanh(self.W_g(xh))             # input candidate: new content
        return f * h + (1.0 - f) * g             # (B, C)


class CfCAttention(nn.Module):
    """
    Drop-in replacement for CausalSelfAttention.

    Processes the sequence step-by-step with a CfC cell, producing one output
    vector per token. The output is projected back to the residual stream.

    Unused arguments (ve, cos_sin, window_size, kv_cache) are accepted for
    interface compatibility with Block.forward but are not used.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.cell = CfCCell(config.n_embd)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        # Must exist for init_weights ve_gate check in GPT
        self.ve_gate = None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()
        h = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            outputs.append(h)
        y = torch.stack(outputs, dim=1)  # (B, T, C)
        return self.c_proj(y)
