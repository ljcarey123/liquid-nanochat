"""
Unit tests for the CfC Liquid cell.

Run: uv run pytest tests/test_liquid.py -v -s
"""
import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.liquid import (
    CfCAttention, CfCCell, CfCCellFull, LiquidHiddenState,
    _cfc_parallel_scan, _cfc_parallel_forward, _cfc_scripted_forward,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def make_model(n_layer=2, n_embd=64, seq_len=32):
    cfg = GPTConfig(
        sequence_len=seq_len,
        vocab_size=256,
        n_layer=n_layer,
        n_head=4,
        n_kv_head=4,
        n_embd=n_embd,
        use_liquid=True,
        mlp_ratio=1,
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=DEVICE)
    model.init_weights()
    return model


# =============================================================================
# CfCCell unit tests
# =============================================================================
class TestCfCCell:
    def test_output_shape(self):
        B, C = 4, 64
        cell = CfCCell(C).to(DEVICE, dtype=DTYPE)
        x = torch.randn(B, C, device=DEVICE, dtype=DTYPE)
        h = torch.zeros(B, C, device=DEVICE, dtype=DTYPE)
        h_new = cell(x, h)
        assert h_new.shape == (B, C)

    def test_no_nan_on_random_input(self):
        B, C = 4, 64
        cell = CfCCell(C).to(DEVICE, dtype=DTYPE)
        x = torch.randn(B, C, device=DEVICE, dtype=DTYPE)
        h = torch.zeros(B, C, device=DEVICE, dtype=DTYPE)
        h_new = cell(x, h)
        assert not torch.isnan(h_new).any(), "CfCCell output contains NaN"

    def test_hidden_state_bounded(self):
        """Output of tanh-gated recurrence must be in (-1, 1)."""
        B, C = 4, 64
        cell = CfCCell(C).to(DEVICE, dtype=DTYPE)
        x = torch.randn(B, C, device=DEVICE, dtype=DTYPE) * 10  # large inputs
        h = torch.zeros(B, C, device=DEVICE, dtype=DTYPE)
        for _ in range(20):
            h = cell(x, h)
        assert h.abs().max().item() <= 1.0 + 1e-4, "Hidden state outside (-1, 1)"

    def test_gradients_flow(self):
        B, C = 2, 32
        cell = CfCCell(C).to(DEVICE, dtype=torch.float32)
        x = torch.randn(B, C, device=DEVICE, requires_grad=True)
        h = torch.zeros(B, C, device=DEVICE)
        h_new = cell(x, h)
        loss = h_new.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_only_two_weight_matrices(self):
        """Option A: CfCCell must have W_f_x and W_g_x only (no h-projections)."""
        cell = CfCCell(32)
        param_names = {name for name, _ in cell.named_parameters()}
        assert "W_f_x.weight" in param_names
        assert "W_g_x.weight" in param_names
        assert "W_f_h.weight" not in param_names, "W_f_h should be removed (Option A)"
        assert "W_g_h.weight" not in param_names, "W_g_h should be removed (Option A)"
        assert len(param_names) == 2


# =============================================================================
# Parallel scan unit tests
# =============================================================================
class TestParallelScan:
    """Verify _cfc_parallel_scan correctness against a reference sequential scan."""

    @staticmethod
    def _sequential_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Reference implementation: h_t = a_t * h_{t-1} + b_t, h_{-1} = 0."""
        B, T, C = a.shape
        h = torch.zeros(B, C, dtype=a.dtype, device=a.device)
        out = torch.empty_like(b)
        for t in range(T):
            h = a[:, t] * h + b[:, t]
            out[:, t] = h
        return out

    def test_matches_sequential_small(self):
        """Parallel scan must match sequential for T=8 (easy to debug)."""
        B, T, C = 2, 8, 16
        a = torch.rand(B, T, C)   # a_t in (0, 1)
        b = torch.randn(B, T, C)
        ref = self._sequential_scan(a, b)
        out = _cfc_parallel_scan(a, b)
        assert torch.allclose(ref, out, atol=1e-5), \
            f"Parallel scan mismatch vs sequential. Max diff: {(ref - out).abs().max():.2e}"

    def test_matches_sequential_t1(self):
        """T=1 edge case: h[0] = b[0] (a[0]*0 + b[0])."""
        B, T, C = 3, 1, 8
        a = torch.rand(B, T, C)
        b = torch.randn(B, T, C)
        ref = self._sequential_scan(a, b)
        out = _cfc_parallel_scan(a, b)
        assert torch.allclose(ref, out, atol=1e-6)

    def test_matches_sequential_non_power_of_two(self):
        """T not a power of 2 (T=13, T=100)."""
        for T in (13, 100):
            B, C = 2, 16
            a = torch.rand(B, T, C)
            b = torch.randn(B, T, C)
            ref = self._sequential_scan(a, b)
            out = _cfc_parallel_scan(a, b)
            assert torch.allclose(ref, out, atol=1e-4), \
                f"T={T}: max diff {(ref - out).abs().max():.2e}"

    def test_no_nan(self):
        B, T, C = 4, 32, 64
        a = torch.rand(B, T, C)
        b = torch.randn(B, T, C)
        out = _cfc_parallel_scan(a, b)
        assert not torch.isnan(out).any()

    def test_causal_independence(self):
        """h[t] must not depend on a[s] or b[s] for s > t."""
        B, T, C = 1, 16, 8
        a = torch.rand(B, T, C)
        b = torch.randn(B, T, C)
        a_mod = a.clone()
        b_mod = b.clone()
        # Perturb positions t >= 8
        a_mod[:, 8:] = torch.rand_like(a_mod[:, 8:])
        b_mod[:, 8:] = torch.randn_like(b_mod[:, 8:])
        out1 = _cfc_parallel_scan(a, b)
        out2 = _cfc_parallel_scan(a_mod, b_mod)
        assert torch.allclose(out1[:, :8], out2[:, :8], atol=1e-6), \
            "Parallel scan is not causal: past outputs changed when future inputs changed"

    def test_parallel_forward_matches_sequential(self):
        """_cfc_parallel_forward must match a sequential reference on the same cell."""
        B, T, C = 2, 16, 32
        cell = CfCCell(C).to(torch.float32)
        x = torch.randn(B, T, C)

        # Parallel
        out_parallel = _cfc_parallel_forward(cell, x)

        # Sequential reference using cell.forward
        with torch.no_grad():
            f_all = torch.sigmoid(cell.W_f_x(x))
            g_all = torch.tanh(cell.W_g_x(x))
        out_seq = self._sequential_scan(f_all, (1.0 - f_all) * g_all)

        assert torch.allclose(out_parallel, out_seq, atol=1e-5), \
            f"_cfc_parallel_forward differs from sequential. Max diff: {(out_parallel - out_seq).abs().max():.2e}"


# =============================================================================
# CfCAttention unit tests
# =============================================================================
class TestCfCAttention:
    def test_output_shape(self):
        B, T, C = 2, 32, 64
        cfg = GPTConfig(sequence_len=T, n_embd=C, use_liquid=True)
        attn = CfCAttention(cfg, layer_idx=0).to(DEVICE, dtype=DTYPE)
        x = torch.randn(B, T, C, device=DEVICE, dtype=DTYPE)
        y = attn(x, ve=None, cos_sin=None, window_size=None, kv_cache=None)
        assert y.shape == (B, T, C)

    def test_no_nan(self):
        B, T, C = 2, 32, 64
        cfg = GPTConfig(sequence_len=T, n_embd=C, use_liquid=True)
        attn = CfCAttention(cfg, layer_idx=0).to(DEVICE, dtype=DTYPE)
        x = torch.randn(B, T, C, device=DEVICE, dtype=DTYPE)
        y = attn(x, ve=None, cos_sin=None, window_size=None, kv_cache=None)
        assert not torch.isnan(y).any()

    def test_causal_independence(self):
        """Output at time t must not depend on tokens after t."""
        B, T, C = 1, 16, 64
        cfg = GPTConfig(sequence_len=T, n_embd=C, use_liquid=True)
        attn = CfCAttention(cfg, layer_idx=0).to(DEVICE, dtype=torch.float32)
        attn.eval()

        x = torch.randn(B, T, C, device=DEVICE)
        x_mod = x.clone()
        x_mod[:, 8:, :] = torch.randn_like(x_mod[:, 8:, :])  # change future tokens

        with torch.no_grad():
            y1 = attn(x, ve=None, cos_sin=None, window_size=None, kv_cache=None)
            y2 = attn(x_mod, ve=None, cos_sin=None, window_size=None, kv_cache=None)

        # Outputs at t < 8 must be identical
        assert torch.allclose(y1[:, :8, :], y2[:, :8, :], atol=1e-5), \
            "CfCAttention is not causal: past outputs changed when future tokens changed"


# =============================================================================
# LiquidHiddenState tests
# =============================================================================
class TestLiquidHiddenState:
    def test_init_zeros(self):
        state = LiquidHiddenState(batch_size=2, n_layer=4, n_embd=64, device="cpu", dtype=torch.float32)
        assert state.h.shape == (4, 2, 64)
        assert state.h.sum() == 0.0

    def test_get_set_hidden(self):
        state = LiquidHiddenState(batch_size=2, n_layer=3, n_embd=16, device="cpu", dtype=torch.float32)
        h = torch.randn(2, 16)
        state.set_hidden(1, h)
        assert torch.equal(state.get_hidden(1), h)
        # Other layers untouched
        assert state.get_hidden(0).sum() == 0.0
        assert state.get_hidden(2).sum() == 0.0

    def test_expand_to_batch(self):
        """expand_to_batch replicates the single-sample hidden state correctly."""
        state = LiquidHiddenState(batch_size=1, n_layer=2, n_embd=8, device="cpu", dtype=torch.float32)
        state.h[0, 0, :] = 1.0
        state.h[1, 0, :] = 2.0

        expanded = state.expand_to_batch(4)
        assert expanded.h.shape == (2, 4, 8)
        # All 4 samples share the same initial values
        assert (expanded.h[0] == 1.0).all()
        assert (expanded.h[1] == 2.0).all()

    def test_expand_requires_batch_size_1(self):
        state = LiquidHiddenState(batch_size=2, n_layer=2, n_embd=8, device="cpu", dtype=torch.float32)
        with pytest.raises(AssertionError):
            state.expand_to_batch(4)

    def test_kvcache_interface(self):
        """get_pos() returns 0, advance() is a no-op."""
        state = LiquidHiddenState(batch_size=1, n_layer=2, n_embd=8, device="cpu", dtype=torch.float32)
        assert state.get_pos() == 0
        state.advance(10)
        assert state.get_pos() == 0  # still 0

    def test_reset(self):
        state = LiquidHiddenState(batch_size=2, n_layer=2, n_embd=8, device="cpu", dtype=torch.float32)
        state.h[:] = 5.0
        state.reset()
        assert state.h.sum() == 0.0


# =============================================================================
# Stateful inference tests
# =============================================================================
class TestStatefulInference:
    """
    Key correctness test: prefill with LiquidHiddenState followed by a
    single-token decode step must produce the same output as running the
    full [prompt + token] sequence in one forward pass.
    """

    def _make_attn(self, C=64):
        cfg = GPTConfig(sequence_len=128, n_embd=C, use_liquid=True)
        attn = CfCAttention(cfg, layer_idx=0).to(torch.float32)
        attn.eval()
        return attn

    def test_decode_step_matches_full_sequence(self):
        """prefill(prompt) + decode(token) == full_seq(prompt + token) at last position."""
        B, T_prompt, C = 1, 8, 64
        attn = self._make_attn(C)

        prompt = torch.randn(B, T_prompt, C)
        token = torch.randn(B, 1, C)
        full_seq = torch.cat([prompt, token], dim=1)  # (B, T_prompt+1, C)

        with torch.no_grad():
            # Reference: full sequence in one pass
            y_full = attn(full_seq, ve=None, cos_sin=None, window_size=None, kv_cache=None)
            y_ref = y_full[:, -1:, :]  # last token output

            # Stateful: prefill then decode
            state = LiquidHiddenState(batch_size=B, n_layer=1, n_embd=C, device="cpu", dtype=torch.float32)
            _ = attn(prompt, ve=None, cos_sin=None, window_size=None, kv_cache=state)
            y_decode = attn(token, ve=None, cos_sin=None, window_size=None, kv_cache=state)

        assert torch.allclose(y_ref, y_decode, atol=1e-5), \
            f"Stateful decode differs from full-sequence. Max diff: {(y_ref - y_decode).abs().max():.2e}"

    def test_multi_step_decode_matches_full_sequence(self):
        """Three decode steps after prefill must match the full sequence at each position."""
        B, T_prompt, C = 1, 6, 64
        attn = self._make_attn(C)
        n_steps = 3

        tokens = torch.randn(B, n_steps, C)
        full_seq = torch.cat([torch.randn(B, T_prompt, C), tokens], dim=1)
        # Re-use the same prompt for both paths
        prompt = full_seq[:, :T_prompt, :]

        with torch.no_grad():
            y_full = attn(full_seq, ve=None, cos_sin=None, window_size=None, kv_cache=None)

            state = LiquidHiddenState(batch_size=B, n_layer=1, n_embd=C, device="cpu", dtype=torch.float32)
            _ = attn(prompt, ve=None, cos_sin=None, window_size=None, kv_cache=state)
            for step in range(n_steps):
                y_step = attn(tokens[:, step:step+1, :], ve=None, cos_sin=None, window_size=None, kv_cache=state)
                y_ref = y_full[:, T_prompt + step:T_prompt + step + 1, :]
                assert torch.allclose(y_ref, y_step, atol=1e-5), \
                    f"Step {step}: max diff {(y_ref - y_step).abs().max():.2e}"

    def test_prefill_stores_correct_layer_hidden_state(self):
        """After prefill, state.h[layer_idx] must equal y[:, -1, :] before c_proj."""
        B, T, C = 2, 10, 32
        cfg = GPTConfig(sequence_len=64, n_embd=C, use_liquid=True)
        attn = CfCAttention(cfg, layer_idx=2).to(torch.float32)
        attn.eval()

        x = torch.randn(B, T, C)
        state = LiquidHiddenState(batch_size=B, n_layer=4, n_embd=C, device="cpu", dtype=torch.float32)

        with torch.no_grad():
            attn(x, ve=None, cos_sin=None, window_size=None, kv_cache=state)
            # The stored hidden state for layer 2 should be the raw parallel scan output
            # at the last position (before c_proj)
            y_raw = _cfc_parallel_forward(attn.cell, x)  # (B, T, C)
            expected_h = y_raw[:, -1, :]
        stored_h = state.get_hidden(2)
        assert torch.allclose(expected_h, stored_h, atol=1e-5)

    def test_expand_and_multi_sample_decode(self):
        """expand_to_batch + decode must give identical outputs for all samples."""
        B_prompt, T_prompt, C = 1, 8, 64
        attn = self._make_attn(C)
        prompt = torch.randn(B_prompt, T_prompt, C)
        token = torch.randn(1, 1, C)

        with torch.no_grad():
            # Prefill with batch=1
            state1 = LiquidHiddenState(batch_size=1, n_layer=1, n_embd=C, device="cpu", dtype=torch.float32)
            _ = attn(prompt, ve=None, cos_sin=None, window_size=None, kv_cache=state1)

            # Expand to 4 samples
            state4 = state1.expand_to_batch(4)
            token4 = token.expand(4, -1, -1)
            y4 = attn(token4, ve=None, cos_sin=None, window_size=None, kv_cache=state4)

        # All 4 rows must be identical (same prompt hidden state)
        assert torch.allclose(y4[0:1], y4[1:2], atol=1e-5)
        assert torch.allclose(y4[0:1], y4[3:4], atol=1e-5)


# =============================================================================
# Full model smoke tests
# =============================================================================
class TestLiquidModel:
    def test_forward_no_nan(self):
        model = make_model()
        model.eval()
        B, T = 2, 32
        x = torch.randint(0, 256, (B, T), device=DEVICE)
        with torch.no_grad():
            loss = model(x, targets=x)
        assert not torch.isnan(loss), f"Forward pass produced NaN loss: {loss}"

    def test_loss_is_finite(self):
        model = make_model()
        model.eval()
        B, T = 2, 32
        x = torch.randint(0, 256, (B, T), device=DEVICE)
        targets = torch.randint(0, 256, (B, T), device=DEVICE)
        with torch.no_grad():
            loss = model(x, targets=targets)
        assert loss.isfinite(), f"Loss is not finite: {loss}"

    def test_overfit_single_batch(self):
        """Loss must decrease monotonically on a single repeated batch."""
        model = make_model(n_layer=2, n_embd=64, seq_len=32)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        B, T = 1, 32
        x = torch.randint(0, 256, (B, T), device=DEVICE)
        targets = x.clone()

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = model(x, targets=targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        # Should drop substantially in 30 steps on a tiny repeated batch
        assert losses[-1] < losses[0] * 0.5, \
            f"Loss barely decreased: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_hidden_state_shapes(self):
        """Verify CfCAttention produces (B, T, C) throughout the model."""
        B, T, C = 2, 16, 64
        cfg = GPTConfig(sequence_len=T, n_embd=C, use_liquid=True, mlp_ratio=1,
                        n_layer=4, n_head=4, n_kv_head=4, vocab_size=256)
        attn_output_shapes = []

        def hook(module, inp, out):
            attn_output_shapes.append(out.shape)

        with torch.device("meta"):
            model = GPT(cfg)
        model.to_empty(device=DEVICE)
        model.init_weights()
        model.eval()

        handles = []
        blocks: torch.nn.ModuleList = model.transformer["h"]  # type: ignore[index]
        for block in blocks:
            handles.append(block.attn.register_forward_hook(hook))  # type: ignore[union-attr]

        x = torch.randint(0, 256, (B, T), device=DEVICE)
        with torch.no_grad():
            model(x)

        for handle in handles:
            handle.remove()

        assert len(attn_output_shapes) == 4
        for shape in attn_output_shapes:
            assert shape == (B, T, C), f"Unexpected attn output shape: {shape}"


# =============================================================================
# Option B: scripted mode tests
# =============================================================================

def make_scripted_model(n_layer=2, n_embd=64, seq_len=32):
    cfg = GPTConfig(
        sequence_len=seq_len, vocab_size=256,
        n_layer=n_layer, n_head=4, n_kv_head=4, n_embd=n_embd,
        use_liquid=True, mlp_ratio=1, liquid_mode="scripted",
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=DEVICE)
    model.init_weights()
    return model


class TestScriptedMode:
    """Option B: full CfC gates compiled with torch.jit.script."""

    def test_cell_full_has_four_weights(self):
        cell = CfCCellFull(32)
        param_names = {name for name, _ in cell.named_parameters()}
        assert {"W_f_x.weight", "W_f_h.weight", "W_g_x.weight", "W_g_h.weight"} == param_names

    def test_cell_full_output_shape(self):
        B, C = 4, 64
        cell = CfCCellFull(C).to(torch.float32)
        x = torch.randn(B, C)
        h = torch.zeros(B, C)
        h_new = cell(x, h)
        assert h_new.shape == (B, C)

    def test_cell_full_hidden_state_bounded(self):
        """Full CfC hidden state must remain in (-1, 1)."""
        B, C = 4, 64
        cell = CfCCellFull(C).to(torch.float32)
        x = torch.randn(B, C) * 10
        h = torch.zeros(B, C)
        for _ in range(20):
            h = cell(x, h)
        assert h.abs().max().item() <= 1.0 + 1e-4

    def test_scripted_forward_output_shape(self):
        B, T, C = 2, 16, 64
        cell = CfCCellFull(C).to(torch.float32)
        x = torch.randn(B, T, C)
        y = _cfc_scripted_forward(cell, x)
        assert y.shape == (B, T, C)

    def test_scripted_forward_no_nan(self):
        B, T, C = 2, 16, 64
        cell = CfCCellFull(C).to(torch.float32)
        x = torch.randn(B, T, C)
        y = _cfc_scripted_forward(cell, x)
        assert not torch.isnan(y).any()

    def test_scripted_forward_causal(self):
        """Output at position t must not depend on positions after t."""
        B, T, C = 1, 16, 32
        cell = CfCCellFull(C).to(torch.float32)
        x = torch.randn(B, T, C)
        x_mod = x.clone()
        x_mod[:, 8:] = torch.randn_like(x_mod[:, 8:])
        with torch.no_grad():
            y1 = _cfc_scripted_forward(cell, x)
            y2 = _cfc_scripted_forward(cell, x_mod)
        assert torch.allclose(y1[:, :8], y2[:, :8], atol=1e-5), \
            "Scripted forward is not causal"

    def test_scripted_matches_python_reference(self):
        """Scripted loop must match a plain Python sequential reference."""
        B, T, C = 2, 12, 32
        cell = CfCCellFull(C).to(torch.float32)
        x = torch.randn(B, T, C)

        # Reference: manual Python loop
        with torch.no_grad():
            fx_all = cell.W_f_x(x)
            gx_all = cell.W_g_x(x)
            h = torch.zeros(B, C)
            ref = torch.zeros(B, T, C)
            for t in range(T):
                f = torch.sigmoid(fx_all[:, t] + cell.W_f_h(h))
                g = torch.tanh(gx_all[:, t] + cell.W_g_h(h))
                h = f * h + (1.0 - f) * g
                ref[:, t] = h

        out = _cfc_scripted_forward(cell, x)
        assert torch.allclose(ref, out, atol=1e-5), \
            f"Scripted forward differs from Python reference. Max diff: {(ref - out).abs().max():.2e}"

    def test_attention_output_shape(self):
        B, T, C = 2, 16, 64
        cfg = GPTConfig(sequence_len=T, n_embd=C, use_liquid=True, liquid_mode="scripted")
        attn = CfCAttention(cfg, layer_idx=0).to(torch.float32)
        x = torch.randn(B, T, C)
        y = attn(x, ve=None, cos_sin=None, window_size=None, kv_cache=None)
        assert y.shape == (B, T, C)

    def test_forward_no_nan(self):
        model = make_scripted_model()
        model.eval()
        x = torch.randint(0, 256, (2, 32), device=DEVICE)
        with torch.no_grad():
            loss = model(x, targets=x)
        assert not torch.isnan(loss)

    def test_overfit_single_batch(self):
        """Scripted mode must also learn on a single repeated batch."""
        model = make_scripted_model(n_layer=2, n_embd=64, seq_len=32)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        x = torch.randint(0, 256, (1, 32), device=DEVICE)
        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            loss = model(x, targets=x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0] * 0.5, \
            f"Scripted mode loss barely decreased: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_stateful_decode_matches_full_sequence(self):
        """Scripted mode: prefill + decode must match full-sequence output at last pos."""
        B, T_prompt, C = 1, 8, 64
        cfg = GPTConfig(sequence_len=128, n_embd=C, use_liquid=True, liquid_mode="scripted")
        attn = CfCAttention(cfg, layer_idx=0).to(torch.float32)
        attn.eval()

        prompt = torch.randn(B, T_prompt, C)
        token = torch.randn(B, 1, C)
        full_seq = torch.cat([prompt, token], dim=1)

        with torch.no_grad():
            y_full = attn(full_seq, ve=None, cos_sin=None, window_size=None, kv_cache=None)
            y_ref = y_full[:, -1:]

            state = LiquidHiddenState(B, 1, C, "cpu", torch.float32)
            attn(prompt, ve=None, cos_sin=None, window_size=None, kv_cache=state)
            y_decode = attn(token, ve=None, cos_sin=None, window_size=None, kv_cache=state)

        assert torch.allclose(y_ref, y_decode, atol=1e-5), \
            f"Scripted stateful decode mismatch. Max diff: {(y_ref - y_decode).abs().max():.2e}"


# =============================================================================
# Cross-mode comparison
# =============================================================================

class TestModeComparison:
    """Sanity checks comparing Option A (parallel) and Option B (scripted)."""

    def test_both_modes_converge(self):
        """Both modes must reduce loss on a single batch within 30 steps."""
        for mode in ("parallel", "scripted"):
            cfg = GPTConfig(
                sequence_len=32, vocab_size=256,
                n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
                use_liquid=True, mlp_ratio=1, liquid_mode=mode,
            )
            with torch.device("meta"):
                model = GPT(cfg)
            model.to_empty(device=DEVICE)
            model.init_weights()
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
            x = torch.randint(0, 256, (1, 32), device=DEVICE)
            losses = []
            for _ in range(30):
                optimizer.zero_grad()
                loss = model(x, targets=x)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            assert losses[-1] < losses[0] * 0.5, \
                f"Mode '{mode}' barely converged: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_both_modes_produce_finite_initial_loss(self):
        """Initial loss for both modes must be finite and in a reasonable range."""
        for mode in ("parallel", "scripted"):
            cfg = GPTConfig(
                sequence_len=32, vocab_size=256,
                n_layer=2, n_head=4, n_kv_head=4, n_embd=64,
                use_liquid=True, mlp_ratio=1, liquid_mode=mode,
            )
            with torch.device("meta"):
                model = GPT(cfg)
            model.to_empty(device=DEVICE)
            model.init_weights()
            model.eval()
            x = torch.randint(0, 256, (2, 32), device=DEVICE)
            with torch.no_grad():
                loss = model(x, targets=x)
            assert loss.isfinite(), f"Mode '{mode}' initial loss not finite: {loss}"
            # Random init should give loss close to log(vocab_size) ≈ 5.5
            assert loss.item() < 10.0, f"Mode '{mode}' suspiciously high initial loss: {loss:.2f}"

    def test_scripted_has_more_parameters_than_parallel(self):
        """Option B has 2× as many cell weights as Option A."""
        def count_cell_params(mode):
            cfg = GPTConfig(sequence_len=32, n_embd=64, use_liquid=True, liquid_mode=mode)
            with torch.device("meta"):
                model = GPT(cfg)
            # Sum parameters in all CfCCell/CfCCellFull modules
            total = 0
            for block in model.transformer["h"]:
                total += sum(p.numel() for p in block.attn.cell.parameters())
            return total

        params_a = count_cell_params("parallel")
        params_b = count_cell_params("scripted")
        assert params_b == 2 * params_a, \
            f"Expected scripted to have 2× parallel cell params. Got {params_a} vs {params_b}"


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "-s"])
