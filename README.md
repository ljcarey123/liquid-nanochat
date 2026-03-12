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
- Handle audio natively — audio is genuinely continuous-time, making CfC a natural fit for speech tokenisation at 75+ tokens/second where O(N) memory is critical

### Architecture change

The only structural modification to nanochat is in [`nanochat/gpt.py`](nanochat/gpt.py): when `GPTConfig(use_liquid=True)` is set, each `Block` swaps its `CausalSelfAttention` for a `CfCAttention` layer (defined in [`nanochat/liquid.py`](nanochat/liquid.py)). Everything else — the tokenizer, dataloader, optimizer, checkpointing, SFT, RL, and chat UI — is unchanged.

```
CfC closed form:
    f = σ( W_f · [x; h] )       ← forget gate
    g = tanh( W_g · [x; h] )    ← input candidate
    h' = f * h + (1 - f) * g    ← new hidden state
```

### Roadmap

| Stage | Goal | Status |
|---|---|---|
| **1 — MVP** | Overfit TinyShakespeare in 100 steps. Smoke test that the Liquid cell learns. | ✅ Done (loss 5.54 → 0.003) |
| **2 — Pretraining** | Train on FineWeb-EDU 100MB subset. Monitor VRAM flatline, loss, tok/s via W&B. | In progress |
| **3 — Post-training** | SFT on SmolTalk via `chat_sft.py`. Deploy with `chat_web.py`. | Not started |
| **4 — Extensions** | Learnable time constants (τ), 4-bit quant, long-context (10K+ tokens), DPO. | Not started |
| **5 — Audio** | Audio tokenisation (EnCodec). ASR fine-tune → speech-in. TTS fine-tune → speech-out. Real-time mic→model→speaker loop. | Not started |

### Quick start (Stage 1 smoke test)

```bash
uv sync --extra gpu
uv run python -m scripts.liquid_smoke_test
```

---

---

## Open Tasks

### Stage 2 — Pretraining

#### T1 · Parallel scan to replace the sequential time loop (critical, performance)

**File:** `nanochat/liquid.py` — `_cfc_sequential_loop`

**Problem:** The `for t in range(T)` loop runs T Python iterations, each with a small `[B, C]→[B, C]` matmul (`W_f_h·h`, `W_g_h·h`). At seq_len=256 and n_embd=256 this is ~530ms/step — too slow for real pretraining.

**Why it can be parallelised:** Once `f_t = sigmoid(W_f_x·x_t)` and `g_t = tanh(W_g_x·x_t)` are precomputed (no `h` dependency), the recurrence reduces to a **linear recurrence**:
```
h_t = a_t * h_{t-1} + b_t
where a_t = sigmoid(W_f_x·x_t + W_f_h·h_{t-1})   ← still h-dependent via W_f_h
```

**Current bottleneck:** `W_f_h` and `W_g_h` — the h-projections inside the gates — create the nonlinear h-dependency. Two options:

- **Option A (drop h-projections from gates):** Set `f_t = sigmoid(W_f_x·x_t)`, `g_t = tanh(W_g_x·x_t)`. The recurrence becomes exactly linear (`a_t, b_t` are precomputed without `h`), enabling a parallel prefix scan in O(log T) steps. This is the approach used by Griffin's RGLRU and Linear RNN architectures. Less expressive than full CfC but fully parallelisable — likely the right trade for pretraining scale.

- **Option B (keep gates, use `torch.jit.script`):** Script the sequential loop with `@torch.jit.script`. The Python overhead disappears but the T sequential CUDA kernel launches remain. Roughly 2× speedup. Keeps full CfC expressivity. Simpler to implement.

**Recommended path:** Implement Option A first (3 lines changed in `CfCCell`), benchmark tok/s and loss quality. Keep Option B as a fallback if quality degrades.

**Reference:** Martin & Cundy (2018) parallel scan; the `associative_scan` in the Mamba S6 codebase.

---

#### T2 · Stateful inference in `engine.py` (critical, inference correctness)

**Files:** `nanochat/engine.py`, `nanochat/gpt.py:384` (`GPT.generate`)

**Problem:** `GPT.generate()` appends a token and re-runs the full forward pass from position 0 each step. For Transformers this is fixed by the KV cache (O(1) per step). For a recurrent model it is even more wasteful — the correct approach is a *stateful step*: run one token through the cell, carry the hidden state `h` forward.

**Current cost:** O(T) per generated token → O(T²) total for a T-token generation. A Transformer with KV cache is O(1) per token.

**What to build:**
1. Add a `LiquidHiddenState` class (analogous to `KVCache`) that stores `h: [B, n_embd]` per layer.
2. Add a `step(x_t, hidden_state)` method to `CfCAttention` that processes a single token and returns the updated hidden state.
3. Modify `GPT.generate()` (or the `Engine` class) to call this step function and thread the hidden state between tokens.

**Result:** O(1) per generated token, constant VRAM regardless of generation length.

---

#### T3 · Bias initialisation for forget gates (correctness / training stability)

**File:** `nanochat/gpt.py` — `init_weights`

**Problem:** The CfC forget gate weights `W_f_x` and `W_f_h` are currently initialised with uniform random values (zero-mean). At initialisation, `sigmoid(≈0) ≈ 0.5`, meaning the model starts forgetting 50% of the hidden state per step. Deep models (many layers, long sequences) will suffer from vanishing information flow at the start of training.

**Fix:** Add a positive bias to `W_f_x` at initialisation so that `sigmoid(b_f) ≈ 0.73–0.88` (corresponding to `b_f ≈ 1.0–2.0`). This ensures the network starts with a high "remember" setting and learns to forget selectively.

**How:** In `init_weights`, after the uniform weight init for liquid blocks, set:
```python
torch.nn.init.constant_(block.attn.cell.W_f_x.bias, 1.0)
```
Note: this requires adding `bias=True` to `W_f_x` only (or using a separate learnable scalar bias per layer). Currently all linears have `bias=False` — a separate `nn.Parameter` scalar per cell is the cleanest approach.

---

#### T4 · Multi-head CfC (expressivity, medium priority)

**File:** `nanochat/liquid.py`

**Background:** The current `CfCAttention` uses a **single CfC cell** operating on the full `n_embd`-dimensional hidden state. Multi-headed attention derives expressivity from parallel subspace processing — each head attends to different aspects of the input. Multi-head CfC applies the same idea to recurrent cells.

**Architecture change:**
```
Current:  1 cell, hidden state h: [B, n_embd]
Proposed: n_head cells, each with h_i: [B, head_dim]  where head_dim = n_embd // n_head
```

Each head runs an independent CfC cell:
```python
# Project input into n_head subspaces
x_heads = x.view(B, T, n_head, head_dim)    # [B, T, n_head, head_dim]

# Per-head weight matrices: W_f, W_g of shape [head_dim, head_dim] each
# Independent hidden state per head: h_heads: [B, n_head, head_dim]

# Concatenate head outputs and project back
output = concat(h_heads, dim=-1).view(B, T, n_embd)
output = c_proj(output)
```

**Implementation notes:**
- Weight matrices shrink from `[n_embd, n_embd]` to `n_head × [head_dim, head_dim]` — same total parameter count, but different inductive bias.
- Heads can be parallelised via batching the `n_head` dimension — no extra sequential loops.
- The hidden state per step is now `[B, n_head, head_dim]`, reshaped to `[B, n_embd]` for storage.
- This is architecturally similar to multi-head linear attention and the RWKV-v6 "multi-head" WKV kernel.

**When to implement:** After the parallel scan (T1) is working. The sequential loop makes benchmarking multi-head vs single-head meaningless until the loop is no longer the bottleneck.

---

#### T5 · W&B monitoring during pretraining (already scaffolded)

**File:** `scripts/liquid_pretrain.py` — W&B calls are already wired.

**What still needs adding:**
- Log `sys/vram_gb` at each step to confirm the O(N) flat VRAM profile (key result for the paper/blog).
- Log `perf/tok_per_sec` computed over a rolling 10-step window.
- Add a `wandb.watch(model)` call with `log_freq=100` to track gradient norms and catch training instability early.
- Log per-layer hidden state norms (`h.norm()`) to verify information is flowing through all layers.

---

### Stage 3 — Post-training

#### T6 · SFT and RL with stateful inference

**Files:** `scripts/chat_sft.py`, `scripts/chat_rl.py`, `nanochat/engine.py`

SFT and RL (GRPO) use the `Engine` class which relies on the KV cache. Once T2 (stateful hidden state inference) is implemented, `Engine` needs a parallel path for Liquid models that maintains `LiquidHiddenState` instead of `KVCache`. The `Engine.generate()` and `Engine.generate_n()` methods both need updating.

---

### Stage 4 — Extensions

#### T7 · Learnable time constants (τ) per cell

**File:** `nanochat/liquid.py`

Currently τ is implicitly 1 (Δt=1, no explicit τ term). In the full LTC/CfC formulation, each cell has a learnable τ (enforced positive via `softplus`) that controls its natural time scale. Longer τ = slower forgetting.

**Implementation:** Add `self.log_tau = nn.Parameter(torch.zeros(n_embd))` to `CfCCell`. In the forward pass, compute `tau = F.softplus(self.log_tau)` and scale the forget gate: `f = sigmoid(gates / tau)`. Initialize `log_tau = 0` so τ starts at `softplus(0) ≈ 0.69`.

**Why it matters:** Different layers naturally learn different time scales (early layers track local syntax, later layers track long-range topic/discourse). Learnable τ lets the model discover this automatically. Expected improvement at seq_len > 512.

---

#### T8 · Long-context evaluation (10K+ tokens)

**Motivation:** The entire point of the Liquid architecture is flat O(N) memory. This needs to be empirically validated against the Transformer baseline.

**What to build:**
1. A `scripts/long_context_eval.py` that ramps seq_len from 256 to 10K and plots VRAM vs length for both Liquid and Transformer models.
2. A perplexity-vs-context-length plot (does liquid model degrade gracefully, or catastrophically, on long contexts?).
3. A throughput plot (tok/s vs seq_len) showing the crossover point where Liquid becomes faster than Transformer.

---

#### T9 · 4-bit quantisation for inference

**Files:** `nanochat/engine.py`, `nanochat/liquid.py`

The recurrent hidden state `h: [B, n_embd]` is tiny compared to a KV cache. The dominant memory cost in inference is the weight matrices. Standard `bitsandbytes` INT4/NF4 quantisation applies directly to `W_f_x`, `W_f_h`, `W_g_x`, `W_g_h`, and `c_proj`. No architecture changes needed — quantisation is a post-training step.

---

### Stage 5 — Audio (Multimodal)

> **Why Liquid is exceptionally suited for audio:** Audio is genuinely continuous-time. A neural audio codec at 24kHz produces ~75 tokens/second — a 10-second utterance is 750 tokens before any text. Transformers at that rate OOM quickly; the Liquid cell's flat O(N) memory profile is the key enabler. The recurrent state also naturally encodes the temporal correlation structure of speech without needing positional embeddings.

> **Prerequisite:** Stage 3 (SFT) should be complete before starting Stage 5. The model needs a working text foundation before multimodal training is attempted.

---

#### T10 · Audio tokeniser integration (EnCodec)

**New file:** `nanochat/audio_tokenizer.py`
**Dependencies:** `encodec` (Meta, pip-installable), `soundfile`, `numpy`

Audio must be converted to discrete token IDs before the model can process it. EnCodec is the recommended codec:
- **Why EnCodec:** Open source (Meta), runs on CPU/GPU, produces discrete tokens at 75 tokens/second (24kHz, single codebook for semantic tokens). The `encodec` Python package installs in seconds.
- **Alternative:** SpeechTokenizer (two-stage: semantic + acoustic codebooks) for higher quality at the cost of complexity. Defer to Stage 5b.

**What to build:**
```python
# nanochat/audio_tokenizer.py
class AudioTokenizer:
    # encode(wav: np.ndarray, sr: int) -> List[int]   # audio → token IDs
    # decode(tokens: List[int]) -> np.ndarray          # token IDs → audio waveform
    # SAMPLE_RATE = 24000
    # TOKENS_PER_SECOND = 75
```

EnCodec's first (semantic) RVQ codebook has 1024 codes. These map cleanly onto new vocabulary entries.

**Vocab expansion:** Add 1024 audio token IDs immediately after the text vocabulary in `GPTConfig.vocab_size`. Special tokens needed: `<|audio_start|>`, `<|audio_end|>`, `<|text_start|>`, `<|text_end|>` (4 more). Total vocab expansion: +1028 IDs. The existing vocabulary padding logic in `GPT.__init__` handles non-round vocab sizes automatically.

**Test:** Encode a 2-second clip → token list → decode back → compare waveforms. Round-trip audio quality with a single codebook is speech-intelligible but not hi-fi — acceptable for this project.

---

#### T11 · Vocabulary expansion and embedding warm-start

**File:** `nanochat/gpt.py` — `GPTConfig`, `GPT.__init__`, `init_weights`

Adding 1028 new token IDs requires a larger embedding table and output projection. The model should be warm-started from the pretrained text checkpoint to preserve language knowledge.

**What to do:**
1. Set `GPTConfig(vocab_size=text_vocab + 1028)` in the audio training config.
2. In `init_weights`, load the pretrained text checkpoint weights for `wte` and `lm_head`, then initialise the new 1028 rows with small random values (same std as the original embedding init, `~0.8`). Do not reinitialise the existing rows.
3. The `lm_head` (unembedding) likewise needs the new columns — initialise them with `std=0.001` (same as original `lm_head` init).

**Checkpoint helper:** Add a `load_partial_checkpoint(path, model)` function to `nanochat/checkpoint_manager.py` that handles mismatched `vocab_size` by copying the matching rows and padding the rest.

---

#### T12 · ASR fine-tuning — speech → text

**New file:** `scripts/audio_sft_asr.py`
**Dataset:** LibriSpeech `train-clean-100` (~28h, freely available via HuggingFace datasets)

Fine-tune the model to transcribe speech: given audio tokens, predict text tokens.

**Sequence format:**
```
<|audio_start|> [audio tokens] <|audio_end|> <|text_start|> [text tokens] <|text_end|>
```

The model sees the full interleaved sequence and is trained with causal LM loss, but loss is **masked to text tokens only** — the model is not penalised for predicting audio continuation.

**Training settings (4GB VRAM):**
- Freeze all layers except `lm_head` and the last 2 `Block` layers for the first 500 steps (avoid destroying text knowledge)
- Then unfreeze all layers with a 10× lower LR for the audio-related parameters
- `seq_len=512` (covers ~5.5s audio + transcription), `device_batch_size=1`, `grad_accum=16`

**Evaluation metric:** Word Error Rate (WER) on LibriSpeech `test-clean`. Target: <30% WER at this scale (not competitive with Whisper, but intelligible).

---

#### T13 · TTS fine-tuning — text → speech

**New file:** `scripts/audio_sft_tts.py`
**Dataset:** LJSpeech (single speaker, 24h, permissive license) or LibriSpeech for multi-speaker

Fine-tune to synthesise speech: given text tokens, predict audio tokens.

**Sequence format:**
```
<|text_start|> [text tokens] <|text_end|> <|audio_start|> [audio tokens] <|audio_end|>
```

Loss is masked to audio tokens only during this fine-tuning stage.

**Why this is hard at small scale:** TTS requires the model to commit to a specific prosody and voice at the first audio token and stay consistent for 75 tokens/second. The Liquid cell's hidden state is well-suited here — it can carry prosodic state across the long audio token sequence without the quadratic memory cost that makes attention-based TTS memory-intensive.

**Key risk:** At this model size (~10M params), the model may produce monotone or robotic speech. That is acceptable for Stage 5 — the goal is audio-out, not broadcast quality.

**Evaluation:** Listen to samples. MOS (Mean Opinion Score) is the gold standard but requires human raters — for this project, spectral similarity (MCD — Mel Cepstral Distortion) is a sufficient automated proxy.

---

#### T14 · Real-time mic → model → speaker inference loop

**New file:** `scripts/audio_chat.py`
**Dependencies:** `sounddevice` (cross-platform mic/speaker I/O), `soundfile`

The full end-to-end spoken conversation loop:

```
[mic] → chunk audio → AudioTokenizer.encode() → prepend to prompt
     → GPT.generate() (stateful, T2 required) → text or audio tokens
     → if audio tokens: AudioTokenizer.decode() → [speaker]
     → if text tokens: print to terminal (ASR output)
```

**Latency considerations:** At 75 audio tokens/second, the model needs to process 75 tokens just to decode 1 second of speech. With stateful inference (T2), each token generation is O(1). The bottleneck shifts to audio decode latency. Target: <500ms first-token latency for spoken responses up to 5s.

**Simplest working version:** Push-to-talk (no streaming), record 3s of audio, transcribe to text, generate text reply, synthesise reply audio, play back. No real-time streaming required for a first working demo.

**Streaming version (stretch):** Use `sounddevice` in callback mode to stream audio tokens directly into the model as they arrive, generating text tokens in parallel. This requires the stateful step inference from T2 and is significantly more complex.

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
