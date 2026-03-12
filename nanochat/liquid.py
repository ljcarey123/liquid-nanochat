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

Speed design (see CfCAttention.forward):
    W_f · [x; h] = W_f_x · x + W_f_h · h
    The x-projections are batched over the full sequence in one matmul;
    only the h-projections remain in the sequential loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from nanochat.common import Linear

if TYPE_CHECKING:
    from nanochat.gpt import GPTConfig


class CfCCell(nn.Module):
    """Single time-step CfC update: (x: [B,C], h: [B,C]) -> h_new: [B,C]

    Weights are split into x-part and h-part so that CfCAttention can
    precompute all x-projections in a single batched matmul.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        # Split W_f = [W_f_x | W_f_h] and W_g = [W_g_x | W_g_h]
        # This is mathematically identical to the original W(cat([x;h])) formulation.
        self.W_f_x = Linear(n_embd, n_embd, bias=False)
        self.W_f_h = Linear(n_embd, n_embd, bias=False)
        self.W_g_x = Linear(n_embd, n_embd, bias=False)
        self.W_g_h = Linear(n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Single-step forward. Prefer step() when x-projections are pre-batched."""
        f = torch.sigmoid(self.W_f_x(x) + self.W_f_h(h))
        g = torch.tanh(self.W_g_x(x) + self.W_g_h(h))
        return f * h + (1.0 - f) * g

    def step(
        self,
        fx: torch.Tensor,
        gx: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """Step using pre-batched x-projections fx = W_f_x(x), gx = W_g_x(x).

        Avoids recomputing the (n_embd→n_embd) x matmul inside the loop.
        """
        f = torch.sigmoid(fx + self.W_f_h(h))
        g = torch.tanh(gx + self.W_g_h(h))
        return f * h + (1.0 - f) * g


def _cfc_sequential_loop(
    cell: CfCCell,
    x: torch.Tensor,
) -> torch.Tensor:
    """Runs the CfC recurrence and returns output tensor (B, T, C).

    Precomputes x-projections for all positions in a single batched matmul,
    then loops only the cheap (B, C)→(B, C) h-projections per step.
    """
    B, T, C = x.shape
    # Batch the input projections over the full sequence — one big matmul each.
    fx_all = cell.W_f_x(x)  # (B, T, C)
    gx_all = cell.W_g_x(x)  # (B, T, C)

    h = x.new_zeros(B, C)
    y = x.new_empty(B, T, C)
    for t in range(T):
        h = cell.step(fx_all[:, t], gx_all[:, t], h)
        y[:, t] = h
    return y


class CfCAttention(nn.Module):
    """
    Drop-in replacement for CausalSelfAttention.

    Processes the sequence step-by-step with a CfC cell, producing one output
    vector per token. The output is projected back to the residual stream.

    Unused arguments (ve, cos_sin, window_size, kv_cache) are accepted for
    interface compatibility with Block.forward but are not used.
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.cell = CfCCell(config.n_embd)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        # Must exist for init_weights ve_gate check in GPT
        self.ve_gate = None

    def forward(
        self,
        x: torch.Tensor,
        ve: object,
        cos_sin: object,
        window_size: object,
        kv_cache: object,
    ) -> torch.Tensor:
        del ve, cos_sin, window_size, kv_cache  # unused; accepted for interface compat with CausalSelfAttention
        y = _cfc_sequential_loop(self.cell, x)
        return self.c_proj(y)
