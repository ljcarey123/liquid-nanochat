"""
Stage 1 smoke test: overfit a tiny Liquid model on a single paragraph of
TinyShakespeare within 100 iterations.

Run from project root:
    uv run python -m scripts.liquid_smoke_test

Success criterion (from LIQUID.md):
    Loss must fall from ~5.5 (random init) to < 1.0 within 100 steps.
"""

import os
import time
import urllib.request
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import COMPUTE_DTYPE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 64
BATCH_SIZE = 1
GRAD_ACCUM = 1          # no accumulation needed for a quick sanity check
NUM_ITERS = 100
LR = 3e-3
TARGET_LOSS = 1.0       # must reach this within NUM_ITERS

# Small liquid model (~300K params)
MODEL_CFG = GPTConfig(
    sequence_len=SEQ_LEN,
    vocab_size=256,         # byte-level: no tokenizer needed
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=128,             # smaller width → faster per-step
    use_liquid=True,
    mlp_ratio=1,            # slim MLP as per LIQUID.md
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare.txt")

def load_shakespeare():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    if not os.path.exists(CACHE):
        print("Downloading TinyShakespeare...")
        urllib.request.urlretrieve(URL, CACHE)
        print(f"  Saved to {CACHE}")
    with open(CACHE, "rb") as f:
        data = f.read()
    return data

# ---------------------------------------------------------------------------
# Overfit batch: always the same paragraph (first SEQ_LEN bytes)
# ---------------------------------------------------------------------------
def get_overfit_batch(data: bytes, device: str):
    chunk = data[:SEQ_LEN + 1]
    x = torch.tensor(list(chunk[:SEQ_LEN]), dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(list(chunk[1:SEQ_LEN + 1]), dtype=torch.long, device=device).unsqueeze(0)
    return x, y

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}  |  Compute dtype: {COMPUTE_DTYPE}")

    data = load_shakespeare()
    print(f"TinyShakespeare: {len(data):,} bytes")
    print(f"Overfit paragraph: {repr(data[:80].decode('utf-8', errors='replace'))}")
    print()

    # Build model
    with torch.device("meta"):
        model = GPT(MODEL_CFG)
    model.to_empty(device=DEVICE)
    model.init_weights()
    model.train()

    nparams = sum(p.numel() for p in model.parameters())
    print(f"Liquid model: {nparams:,} parameters")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    print(f"{'Step':>5}  {'Loss':>8}  {'ms/step':>8}  {'ETA':>8}  {'Progress'}")
    print("-" * 65)

    step_times: list[float] = []
    prev_loss: float | None = None
    accum_loss = 0.0

    for step in range(1, NUM_ITERS + 1):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            x, y = get_overfit_batch(data, DEVICE)   # same paragraph every time
            loss = model(x, targets=y)
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item() / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_ms = (time.perf_counter() - t0) * 1000
        step_times.append(step_ms)
        avg_ms = sum(step_times[-10:]) / len(step_times[-10:])  # rolling average
        remaining_steps = NUM_ITERS - step
        eta_s = avg_ms * remaining_steps / 1000

        if step % 10 == 0 or step == 1:
            bar_filled = int(step / NUM_ITERS * 20)
            bar = "#" * bar_filled + "." * (20 - bar_filled)
            delta = f"({accum_loss - prev_loss:+.4f})" if prev_loss is not None else ""
            eta_str = f"{eta_s:.0f}s" if eta_s < 60 else f"{eta_s/60:.1f}m"
            print(f"{step:>5}  {accum_loss:>8.4f}  {avg_ms:>8.1f}  {eta_str:>8}  [{bar}] {step}/{NUM_ITERS}  {delta}")
            prev_loss = accum_loss

    print()
    final_loss = accum_loss
    if final_loss < TARGET_LOSS:
        print(f"PASSED — loss {final_loss:.4f} < {TARGET_LOSS} target")
    else:
        print(f"FAILED — loss {final_loss:.4f} did not reach {TARGET_LOSS} target")

    # Quick generation sample
    print("\n--- Generation sample (greedy) ---")
    model.eval()
    prompt = list(data[:32])
    generated = [int(t) for t in model.generate(prompt, max_tokens=128, temperature=0.8)]
    text = bytes(prompt + generated).decode("utf-8", errors="replace")
    print(repr(text))


if __name__ == "__main__":
    main()
