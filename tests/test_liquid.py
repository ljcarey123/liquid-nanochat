"""
Unit tests for the CfC Liquid cell.

Run: uv run pytest tests/test_liquid.py -v -s
"""
import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.liquid import CfCAttention, CfCCell

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


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v", "-s"])
