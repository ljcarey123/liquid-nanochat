# CLAUDE.md — Liquid NanoChat

## Project Summary

This is a fork of `nanochat` (a minimal full-stack ChatGPT clone) with the goal of replacing the O(N²) attention mechanism with a **Closed-form Continuous-time (CfC) Liquid cell** — a recurrent architecture that operates in O(N) memory. The full pipeline (pretraining → SFT → RL → chat deployment) is preserved.

**Why Liquid?** CfC cells carry a hidden state across the sequence instead of growing a KV cache, making inference memory flat regardless of context length. This makes the full training cycle feasible on a single 4GB laptop GPU.

---

## Hardware

- **GPU**: NVIDIA GeForce RTX 3050 Laptop (4096 MiB VRAM, SM 8.6 Ampere)
- **Driver**: 595.79 / CUDA 13.2 (supports cu128 PyTorch wheels)
- **bfloat16**: Supported (Ampere+)
- **Constraint**: 4GB VRAM — keep batch sizes small, use fp16 or bf16

---

## Environment

**Package manager**: `uv` (never use pip directly)

**Virtual environment** lives at `.venv/` in the project root. Always activate or prefix commands with `uv run`:

```bash
# Activate
source .venv/Scripts/activate        # Windows (bash/git bash)
# or prefix every command:
uv run python scripts/base_train.py
```

**Install**:
```bash
uv sync --extra gpu          # GPU (CUDA 12.8)
uv sync --extra cpu          # CPU-only fallback
uv sync --group dev          # Include dev/test deps
```

**Run tests**:
```bash
uv run pytest
```

---

## Key Files

| File | Role |
|---|---|
| `nanochat/gpt.py` | **Core model** — GPT transformer, `CausalSelfAttention`, `Block`, `GPT` |
| `nanochat/engine.py` | Inference engine with KV cache; needs redesign for stateful Liquid inference |
| `nanochat/common.py` | Dtype management (`COMPUTE_DTYPE`), distributed setup, GPU peak FLOP table |
| `nanochat/flash_attention.py` | FA3/SDPA abstraction — may be bypassed by Liquid cell |
| `nanochat/optim.py` | MuonAdamW hybrid optimizer |
| `nanochat/dataloader.py` | BOS-aligned best-fit packing for distributed training |
| `nanochat/checkpoint_manager.py` | Checkpoint save/load, model reconstruction |
| `scripts/base_train.py` | Pretraining entrypoint |
| `scripts/chat_sft.py` | Supervised finetuning |
| `scripts/chat_rl.py` | GRPO reinforcement learning |
| `scripts/chat_web.py` | FastAPI web UI server |
| `tasks/` | Evaluation tasks (GSM8K, MMLU, ARC, HumanEval, SmolTalk) |
| `runs/speedrun.sh` | Full 3-hour pipeline on 8×H100 |

---

## Architecture Overview

### Current: Transformer Block (`gpt.py`)

```
Input x
  └─> LayerNorm → CausalSelfAttention → residual  (with resid_lambda + x0_lambda)
  └─> LayerNorm → MLP (ReLU²)         → residual
```

**CausalSelfAttention** key features:
- Rotary positional embeddings (no learnable pos embeds)
- QK normalization for training stability
- Group-Query Attention (GQA) — fewer KV heads than Q heads
- Value Embeddings (ResFormer-style, per-layer)
- Sliding window attention (`window_pattern` e.g. `"SSSL"`)
- Flash Attention 3 on Hopper (SM90), SDPA elsewhere

**GPTConfig** tuning parameters:
- `n_layer` (depth), `n_embd` (hidden dim via aspect_ratio × depth)
- `n_head`, `n_kv_head` for GQA
- `sequence_len`, `window_pattern`

### Target: Liquid Block

Replace `CausalSelfAttention` with a **CfC cell** that:
- Maintains a hidden state `h` across the sequence (shape: `[B, n_embd]`)
- Uses a continuous-time ODE solved in closed form: `h(t+1) = f(h(t), x(t), τ)`
- No attention matrix — O(N) memory, O(N) compute
- Preserves the `Block` interface so `base_train.py` needs zero changes

New file to create: `nanochat/liquid.py` — contains `CfCCell` and `LiquidBlock`.

---

## Roadmap

| Stage | Goal | Status |
|---|---|---|
| **1 — MVP** | Replace attention with CfC cell; overfit TinyShakespeare in 100 iterations | Not started |
| **2 — Pretraining** | Train on FineWeb-EDU 100MB subset; W&B monitoring | Not started |
| **3 — Post-training** | SFT on SmolTalk, deploy with chat_web.py | Not started |
| **4 — Extensions** | Learnable τ, 4-bit quant, long-context (10K+), DPO | Not started |

**Stage 1 smoke-test settings** (fits in 4GB):
```
device_batch_size: 1
gradient_accumulation_steps: 16
dtype: float16
dataset: TinyShakespeare (~1MB)
sequence_len: 256
```

---

## Coding Guidelines

### General
- **Minimal changes** — only touch what is necessary for the current stage. Do not refactor unrelated code.
- **Preserve interfaces** — the `Block` and `GPT` class signatures must stay compatible with `base_train.py`, `engine.py`, and checkpointing.
- **No new files unless necessary** — prefer adding to existing files (e.g. `gpt.py`) before creating new ones.
- **No backwards-compatibility shims** — if something is replaced, replace it cleanly.
- **No docstrings or comments on untouched code** — only comment novel Liquid-specific logic.

### PyTorch
- Use explicit dtypes (`COMPUTE_DTYPE` from `common.py`), never `torch.amp.autocast`.
- No model biases — consistent with existing architecture.
- Master weights stay fp32 for optimizer; cast in forward pass only.
- Use `torch.compile` where possible for speed on SM 8.6.
- Avoid in-place operations on tensors that require grad.

### Liquid / CfC Specifics
- Hidden state `h` must be detached between batches (not between time steps within a sequence).
- Time step Δt is fixed at 1.0 for language modelling (no irregular time).
- Gate activations: sigmoid for forget/input gates, tanh for cell content.
- Initialize τ (time constant) to 1.0; make learnable in Stage 4.
- The CfC closed-form: `h = sigmoid(f) * h_prev + (1 - sigmoid(f)) * tanh(g)` where `f`, `g` are linear projections of `[x, h_prev]`.

### Testing
- Run `uv run pytest` before committing architecture changes.
- Smoke test: model forward pass must not produce NaN on random input before any training.
- Overfit test: loss must decrease monotonically on a single repeated batch.

### Memory Targets (4GB VRAM)
- Peak VRAM < 3.5GB during training (leave headroom for OS/driver)
- Use `torch.cuda.max_memory_allocated()` to verify after each new configuration
- If OOM: reduce `device_batch_size` first, then `sequence_len`, then model depth

---

## Notes

- `engine.py` uses a `KVCache` that pre-allocates `[B, T, H, D]` tensors. For Liquid inference, this becomes a hidden-state cache `[B, n_embd]` — much smaller.
- The `flash_attention.py` abstraction is not needed for Liquid cells (no attention matrix).
- `fp8.py` is H100-specific (SM90) — ignore for RTX 3050 (SM 8.6).
- `runs/speedrun.sh` targets 8×H100; single-GPU training uses `python -m scripts.base_train` directly.
