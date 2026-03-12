# liquid-nanochat

> A fork of [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy, extended to explore **Liquid Neural Networks** on consumer hardware.

---

## What this fork is about

**liquid-nanochat** replaces the O(N²) Transformer attention mechanism with a **Closed-form Continuous-time (CfC) Liquid cell** — a recurrent architecture that maintains a hidden state across the sequence in O(N) memory. The goal is to demonstrate that the full LLM training cycle (pretraining → SFT → RL → chat deployment) can run on a single laptop GPU (4GB VRAM) using a memory-efficient recurrent architecture.

### Why Liquid?

Standard Transformers scale memory quadratically with context length because of the attention matrix and KV cache. A CfC Liquid cell replaces all of that with a single hidden state vector of fixed size — memory stays flat no matter how long the sequence grows. This makes it possible to:

- Train on a 4GB GPU where a standard Transformer would OOM
- Run inference at constant memory, regardless of context length
- Explore continuous-time neural dynamics in the context of language modelling

### Architecture change

The only structural modification to nanochat is in [`nanochat/gpt.py`](nanochat/gpt.py): when `GPTConfig(use_liquid=True)` is set, each `Block` swaps its `CausalSelfAttention` for a `CfCAttention` layer (defined in [`nanochat/liquid.py`](nanochat/liquid.py)). Everything else — the tokenizer, dataloader, optimizer, checkpointing, SFT, RL, and chat UI — is unchanged.

```
CfC closed form:
    f = σ( W_f · [x; h] )       ← forget gate
    g = tanh( W_g · [x; h] )    ← input candidate
    h' = f * h + (1 - f) * g    ← new hidden state
```

### Roadmap

| Stage | Goal |
|---|---|
| **1 — MVP** | Overfit TinyShakespeare in 100 steps. Smoke test that the Liquid cell learns. |
| **2 — Pretraining** | Train on FineWeb-EDU 100MB subset. Monitor VRAM flatline, loss, tok/s via W&B. |
| **3 — Post-training** | SFT on SmolTalk via `chat_sft.py`. Deploy with `chat_web.py`. |
| **4 — Extensions** | Learnable time constants (τ), 4-bit quant, long-context (10K+ tokens), DPO. |

### Quick start (Stage 1 smoke test)

```bash
uv sync --extra gpu
uv run python -m scripts.liquid_smoke_test
```

---

## Original nanochat

![nanochat logo](dev/nanochat.png)

nanochat by Andrej Karpathy is the simplest experimental harness for training LLMs end-to-end — tokenization, pretraining, finetuning, evaluation, inference, and a chat UI — on a single GPU node. The code is minimal, hackable, and designed to be forked. This project builds directly on that foundation.

## Precision / dtype

nanochat does not use `torch.amp.autocast`. Instead, precision is managed explicitly through a single global `COMPUTE_DTYPE` (defined in `nanochat/common.py`). By default this is auto-detected based on your hardware:

| Hardware | Default dtype | Why |
|----------|--------------|-----|
| CUDA SM 80+ (A100, H100, RTX 30xx+) | `bfloat16` | Native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | No bf16; fp16 available via `NANOCHAT_DTYPE=float16` |
| CPU / MPS | `float32` | No reduced-precision tensor cores |

You can override the default with the `NANOCHAT_DTYPE` environment variable:

```bash
NANOCHAT_DTYPE=float16 uv run python -m scripts.liquid_smoke_test
```

## File structure

```
.
├── nanochat/
│   ├── gpt.py                      # GPT transformer + Liquid config flags
│   ├── liquid.py                   # CfC Liquid cell (new)
│   ├── engine.py                   # Inference with KV cache
│   ├── common.py                   # COMPUTE_DTYPE, distributed utils
│   ├── dataloader.py               # Distributed data loader
│   ├── dataset.py                  # Data download/preprocessing
│   ├── optim.py                    # MuonAdamW optimizer
│   ├── tokenizer.py                # BPE tokenizer
│   ├── checkpoint_manager.py       # Checkpoint save/load
│   └── ...
├── scripts/
│   ├── liquid_smoke_test.py        # Stage 1: overfit TinyShakespeare (new)
│   ├── base_train.py               # Base pretraining
│   ├── chat_sft.py                 # Supervised finetuning
│   ├── chat_rl.py                  # Reinforcement learning (GRPO)
│   └── chat_web.py                 # Chat web UI
├── tasks/                          # Evaluation tasks (GSM8K, MMLU, ARC, ...)
├── CLAUDE.md                       # Project guidelines for AI-assisted development
└── pyproject.toml
```

## Acknowledgements

- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy — the entire pipeline this fork is built on.
- [nanoGPT](https://github.com/karpathy/nanoGPT) — the original minimal GPT project.
- [Closed-form Continuous-time Neural Networks](https://arxiv.org/abs/2106.13898) (Hasani et al., 2022) — the CfC architecture implemented here.
- [HuggingFace](https://huggingface.co/) for datasets and tokenizer tooling.

## License

MIT
