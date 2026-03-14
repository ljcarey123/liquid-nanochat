"""
Closed-form Continuous-time (CfC) Liquid cell.

Drop-in replacement for CausalSelfAttention. Instead of building an O(N²)
attention matrix, each token updates a hidden state h via a gated ODE solved
in closed form — O(N) memory, O(N) compute.

Two implementation modes selected via GPTConfig.liquid_mode:

  "parallel"  — Option A: input-only gates (no h-dependency).
        f = σ(W_f_x·x),  g = tanh(W_g_x·x)
        Gates are precomputed for all positions in one batched matmul, then
        a Hillis-Steele parallel prefix scan solves h_t = f_t*h_{t-1} + b_t
        in O(log T) sequential steps.  Fully vectorised; no Python loop.
        Trade-off: gates cannot adapt to current hidden state (less expressive
        than full CfC, but matches Griffin/RWKV/Mamba-style gated linear RNNs).

  "scripted"  — Option B: full CfC gates (h-dependency preserved).
        f = σ(W_f_x·x + W_f_h·h),  g = tanh(W_g_x·x + W_g_h·h)
        The sequential recurrence is compiled with torch.jit.script, removing
        Python loop overhead while keeping all T kernel launches.
        ~2× faster than an unscripted loop; full CfC expressivity retained.

Both modes share the same LiquidHiddenState cache and stateful decode path
(O(1) memory per generated token regardless of generation length).

Reference: https://arxiv.org/abs/2106.13898 (Hasani et al., 2022)
Parallel scan: Hillis & Steele (1986) prefix-scan algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import torch.nn as nn

from nanochat.common import Linear

if TYPE_CHECKING:
    from nanochat.gpt import GPTConfig


# =============================================================================
# CfC cells
# =============================================================================

class CfCCell(nn.Module):
    """Option A cell: gates are functions of x only (no h-dependency).

    Enables the parallel prefix scan — no sequential loop over T.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.W_f_x = Linear(n_embd, n_embd, bias=False)  # forget gate (input)
        self.W_g_x = Linear(n_embd, n_embd, bias=False)  # candidate  (input)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Single-step update used for stateful decode (T=1).
        x: (B, C),  h: (B, C)  ->  h_new: (B, C)
        """
        f = torch.sigmoid(self.W_f_x(x))
        g = torch.tanh(self.W_g_x(x))
        return f * h + (1.0 - f) * g


class CfCCellFull(nn.Module):
    """Option B cell: full CfC gates that depend on both x and h.

    More expressive than CfCCell; requires the scripted sequential loop.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.W_f_x = Linear(n_embd, n_embd, bias=False)  # forget gate (input)
        self.W_f_h = Linear(n_embd, n_embd, bias=False)  # forget gate (state)
        self.W_g_x = Linear(n_embd, n_embd, bias=False)  # candidate  (input)
        self.W_g_h = Linear(n_embd, n_embd, bias=False)  # candidate  (state)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Single-step update used for stateful decode (T=1).
        x: (B, C),  h: (B, C)  ->  h_new: (B, C)
        """
        f = torch.sigmoid(self.W_f_x(x) + self.W_f_h(h))
        g = torch.tanh(self.W_g_x(x) + self.W_g_h(h))
        return f * h + (1.0 - f) * g


# =============================================================================
# Option A: parallel prefix scan
# =============================================================================

def _cfc_parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve h_t = a_t * h_{t-1} + b_t, h_{-1} = 0, for all t in [0, T).

    Uses the Hillis-Steele parallel prefix scan. Each position t tracks the
    composite linear map (A[t], B[t]) such that h[t] = A[t]*h_init + B[t].
    Since h_init = 0, the final h[t] = B[t] after the scan completes.

    Composition operator: (A_r, B_r) ∘ (A_l, B_l) = (A_r*A_l, A_r*B_l + B_r)

    Complexity: O(T log T) work, O(log T) sequential kernel launches.

    Args:
        a: (B, T, C) — per-step carry coefficients (forget gates)
        b: (B, T, C) — per-step additive terms
    Returns:
        h: (B, T, C) — all hidden states
    """
    _, T, _ = a.shape
    A = a.clone()
    B_acc = b.clone()
    stride = 1
    while stride < T:
        A_prev = A.clone()
        B_prev = B_acc.clone()
        A[:, stride:] = A_prev[:, stride:] * A_prev[:, :T - stride]
        B_acc[:, stride:] = A_prev[:, stride:] * B_prev[:, :T - stride] + B_prev[:, stride:]
        stride <<= 1
    return B_acc


def _cfc_parallel_forward(cell: CfCCell, x: torch.Tensor) -> torch.Tensor:
    """Option A full-sequence forward via parallel scan. Returns h: (B, T, C).

    Two batched matmuls over the full sequence, then O(log T) scan passes.
    """
    f = torch.sigmoid(cell.W_f_x(x))   # (B, T, C)
    g = torch.tanh(cell.W_g_x(x))      # (B, T, C)
    return _cfc_parallel_scan(f, (1.0 - f) * g)


# =============================================================================
# Option B: torch.jit.script sequential loop (full CfC)
# =============================================================================

@torch.jit.script
def _cfc_scripted_loop_impl(
    W_f_x_w: torch.Tensor,
    W_f_h_w: torch.Tensor,
    W_g_x_w: torch.Tensor,
    W_g_h_w: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """TorchScript inner loop for Option B — eliminates Python overhead.

    Takes weight tensors directly (not nn.Module) so the function is
    fully scriptable. All T steps still execute sequentially, but without
    Python interpreter overhead between kernel launches.
    """
    B = x.shape[0]
    T = x.shape[1]
    fx_all = x @ W_f_x_w.t()   # (B, T, C) — batched over full sequence
    gx_all = x @ W_g_x_w.t()   # (B, T, C)
    h = x.new_zeros(B, x.shape[2])
    hs: List[torch.Tensor] = []
    for t in range(T):
        f = torch.sigmoid(fx_all[:, t] + h @ W_f_h_w.t())
        g = torch.tanh(gx_all[:, t] + h @ W_g_h_w.t())
        h = f * h + (1.0 - f) * g
        hs.append(h)
    return torch.stack(hs, dim=1)  # (B, T, C)


def _cfc_scripted_forward(cell: CfCCellFull, x: torch.Tensor) -> torch.Tensor:
    """Option B full-sequence forward using the scripted loop.

    Weights are cast to x.dtype so the scripted matmul kernels get consistent
    dtypes (nn.Linear handles this implicitly; raw @ in TorchScript does not).
    The cast is a no-op when dtypes already match (fp32 training path).
    """
    dtype = x.dtype
    return _cfc_scripted_loop_impl(
        cell.W_f_x.weight.to(dtype=dtype), cell.W_f_h.weight.to(dtype=dtype),
        cell.W_g_x.weight.to(dtype=dtype), cell.W_g_h.weight.to(dtype=dtype),
        x,
    )


# =============================================================================
# LiquidHiddenState — shared by both modes
# =============================================================================

class LiquidHiddenState:
    """Hidden state cache for Liquid (CfC) models — the stateful analogue of KVCache.

    Stores h: (n_layer, B, n_embd) — one hidden vector per layer per batch element.
    Memory footprint is tiny: n_layer * B * n_embd floats vs the KVCache's
    n_layer * B * T * n_kv_head * head_dim floats.

    Implements the same get_pos() / advance() interface as KVCache so that
    GPT.forward() can call them without knowing which cache type is in use.
    """

    def __init__(self, batch_size: int, n_layer: int, n_embd: int, device, dtype):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.h = torch.zeros(n_layer, batch_size, n_embd, device=device, dtype=dtype)

    def get_hidden(self, layer_idx: int) -> torch.Tensor:
        """Return hidden state for layer_idx: (B, n_embd)."""
        return self.h[layer_idx]

    def set_hidden(self, layer_idx: int, h: torch.Tensor) -> None:
        """Store updated hidden state for layer_idx."""
        self.h[layer_idx] = h

    def reset(self) -> None:
        self.h.zero_()

    def expand_to_batch(self, batch_size: int) -> LiquidHiddenState:
        """Replicate a batch_size=1 hidden state to batch_size=N.

        Used after single-sample prefill to fan out for multi-sample decoding,
        analogous to KVCache.prefill() which copies cached KV tensors.
        """
        assert self.h.shape[1] == 1, f"expand_to_batch requires batch_size=1, got {self.h.shape[1]}"
        new = LiquidHiddenState.__new__(LiquidHiddenState)
        new.n_layer = self.n_layer
        new.n_embd = self.n_embd
        new.h = self.h.expand(-1, batch_size, -1).clone()
        return new

    # ---- KVCache interface compatibility ----

    def get_pos(self) -> int:
        """Liquid does not use positional offsets — always 0."""
        return 0

    def advance(self, n: int) -> None:
        """No-op: Liquid carries full context in h, not a position counter."""
        pass


# =============================================================================
# CfCAttention — drop-in for CausalSelfAttention, supports both modes
# =============================================================================

class CfCAttention(nn.Module):
    """
    Drop-in replacement for CausalSelfAttention.

    Processes the sequence with a CfC cell, producing one output vector per
    token. The output is projected back to the residual stream.

    Mode is selected at construction time via config.liquid_mode:
      "parallel" — Option A: parallel scan, no h-dependency in gates.
      "scripted" — Option B: torch.jit.script loop, full CfC gates.

    Both modes share the same stateful inference path:
      Prefill  (kv_cache is LiquidHiddenState, T > 1): parallel/scripted forward,
               final hidden state h[:, -1, :] stored in cache.
      Decode   (kv_cache is LiquidHiddenState, T == 1): single cell.forward step,
               O(1) per token, constant VRAM regardless of generation length.
      Training (kv_cache is None): parallel/scripted forward, no state storage.

    Unused arguments (ve, cos_sin, window_size) are accepted for interface
    compatibility with Block.forward.
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.liquid_mode = getattr(config, 'liquid_mode', 'parallel')
        if self.liquid_mode == "scripted":
            self.cell: nn.Module = CfCCellFull(config.n_embd)
        else:
            self.cell = CfCCell(config.n_embd)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate = None  # required for init_weights ve_gate check in GPT

    def _full_sequence_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.liquid_mode == "scripted":
            assert isinstance(self.cell, CfCCellFull)
            return _cfc_scripted_forward(self.cell, x)
        else:
            assert isinstance(self.cell, CfCCell)
            return _cfc_parallel_forward(self.cell, x)

    def forward(
        self,
        x: torch.Tensor,
        ve: object,
        cos_sin: object,
        window_size: object,
        kv_cache: object,
    ) -> torch.Tensor:
        del ve, cos_sin, window_size  # unused; accepted for interface compat

        if isinstance(kv_cache, LiquidHiddenState):
            if x.shape[1] == 1:
                # Stateful single-token decode: O(1) per step, both modes
                h = kv_cache.get_hidden(self.layer_idx)   # (B, C)
                h_new = self.cell(x[:, 0, :], h)          # (B, C)
                kv_cache.set_hidden(self.layer_idx, h_new)
                y = h_new.unsqueeze(1)                     # (B, 1, C)
            else:
                # Prefill: run full sequence, store final hidden state
                y = self._full_sequence_forward(x)                     # (B, T, C)
                kv_cache.set_hidden(self.layer_idx, y[:, -1, :].detach())
        else:
            y = self._full_sequence_forward(x)

        return self.c_proj(y)
