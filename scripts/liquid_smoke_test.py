"""
Stage 1 smoke test: overfit a tiny Liquid model on a single paragraph of
TinyShakespeare within 100 iterations.

Runs both liquid_mode="parallel" (Option A) and liquid_mode="scripted" (Option B)
and prints a side-by-side comparison of loss, speed, and a generation sample.

Run from project root:
    uv run python -m scripts.liquid_smoke_test          # compare both modes
    uv run python -m scripts.liquid_smoke_test parallel  # Option A only
    uv run python -m scripts.liquid_smoke_test scripted  # Option B only

Success criterion:
    Loss must fall from ~5.5 (random init) to < 1.0 within 100 steps.
"""

import os
import sys
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
NUM_ITERS = 100
LR = 3e-3
TARGET_LOSS = 1.0

BASE_CFG = dict(
    sequence_len=SEQ_LEN,
    vocab_size=256,
    n_layer=4,
    n_head=4,
    n_kv_head=4,
    n_embd=128,
    use_liquid=True,
    mlp_ratio=1,
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
        return f.read()

def get_batch(data: bytes):
    chunk = data[:SEQ_LEN + 1]
    x = torch.tensor(list(chunk[:SEQ_LEN]), dtype=torch.long, device=DEVICE).unsqueeze(0)
    y = torch.tensor(list(chunk[1:SEQ_LEN + 1]), dtype=torch.long, device=DEVICE).unsqueeze(0)
    return x, y

# ---------------------------------------------------------------------------
# Single training run — returns (losses, step_times, model)
# ---------------------------------------------------------------------------
def run_mode(mode: str, data: bytes) -> tuple[list[float], list[float], GPT]:
    cfg = GPTConfig(**BASE_CFG, liquid_mode=mode)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=DEVICE)
    model.init_weights()
    model.train()

    nparams = sum(p.numel() for p in model.parameters())
    cell_params = sum(
        p.numel()
        for block in model.transformer["h"] # type: ignore[index]
        for p in block.attn.cell.parameters()
    )
    print(f"  [{mode:8}]  {nparams:,} total params  ({cell_params:,} cell params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    losses: list[float] = []
    step_times: list[float] = []
    x, y = get_batch(data)

    for step in range(1, NUM_ITERS + 1):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_ms = (time.perf_counter() - t0) * 1000
        losses.append(loss.item())
        step_times.append(step_ms)

    return losses, step_times, model

# ---------------------------------------------------------------------------
# Pretty progress table (printed live for a single mode)
# ---------------------------------------------------------------------------
def run_mode_verbose(mode: str, data: bytes) -> tuple[list[float], list[float], GPT]:
    cfg = GPTConfig(**BASE_CFG, liquid_mode=mode)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=DEVICE)
    model.init_weights()
    model.train()

    nparams = sum(p.numel() for p in model.parameters())
    print(f"Liquid model ({mode}): {nparams:,} parameters")
    print()
    print(f"{'Step':>5}  {'Loss':>8}  {'ms/step':>8}  {'ETA':>8}  {'Progress'}")
    print("-" * 65)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    losses: list[float] = []
    step_times: list[float] = []
    prev_loss = None
    x, y = get_batch(data)

    for step in range(1, NUM_ITERS + 1):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_ms = (time.perf_counter() - t0) * 1000
        losses.append(loss.item())
        step_times.append(step_ms)

        if step % 10 == 0 or step == 1:
            avg_ms = sum(step_times[-10:]) / min(len(step_times), 10)
            eta_s = avg_ms * (NUM_ITERS - step) / 1000
            eta_str = f"{eta_s:.0f}s" if eta_s < 60 else f"{eta_s/60:.1f}m"
            bar = "#" * int(step / NUM_ITERS * 20) + "." * (20 - int(step / NUM_ITERS * 20))
            delta = f"({loss.item() - prev_loss:+.4f})" if prev_loss is not None else ""
            print(f"{step:>5}  {loss.item():>8.4f}  {avg_ms:>8.1f}  {eta_str:>8}  [{bar}] {step}/{NUM_ITERS}  {delta}")
            prev_loss = loss.item()

    return losses, step_times, model

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    modes_arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    modes = ["parallel", "scripted"] if modes_arg == "both" else [modes_arg]

    print(f"Device: {DEVICE}  |  Compute dtype: {COMPUTE_DTYPE}")
    print()

    data = load_shakespeare()
    print(f"TinyShakespeare: {len(data):,} bytes")
    print(f"Overfit paragraph: {repr(data[:80].decode('utf-8', errors='replace'))}")
    print()

    results: dict[str, tuple[list[float], list[float], GPT]] = {}

    if len(modes) == 1:
        # Single mode: verbose live output
        mode = modes[0]
        print(f"=== {mode.upper()} mode ===")
        losses, step_times, model = run_mode_verbose(mode, data)
        results[mode] = (losses, step_times, model)
    else:
        # Both modes: run silently, then print comparison
        print("Running both modes (silent)...")
        print()
        for mode in modes:
            losses, step_times, model = run_mode(mode, data)
            results[mode] = (losses, step_times, model)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print()
    print("=" * 65)
    print(f"{'':20}  {'parallel (A)':>18}  {'scripted (B)':>18}")
    print("-" * 65)

    ROWS = [1, 10, 25, 50, 75, 100]
    for step in ROWS:
        row = f"  loss @ step {step:<5}"
        for mode in ("parallel", "scripted"):
            if mode in results:
                val = f"{results[mode][0][step - 1]:.4f}"
            else:
                val = "  —   "
            row += f"  {val:>18}"
        print(row)

    print("-" * 65)
    for mode in ("parallel", "scripted"):
        if mode not in results:
            continue
        losses, step_times, _ = results[mode]
        avg_ms = sum(step_times[5:]) / max(len(step_times[5:]), 1)  # skip warmup
        status = "PASSED" if losses[-1] < TARGET_LOSS else "FAILED"
        print(f"  {mode:10}  final loss: {losses[-1]:.4f}  avg ms/step: {avg_ms:.1f}  {status}")

    # ---------------------------------------------------------------------------
    # Generation samples
    # ---------------------------------------------------------------------------
    print()
    print("=" * 65)
    print("Generation samples (greedy, 128 tokens from same prompt):")
    prompt = list(data[:32])
    for mode, (_, _, model) in results.items():
        model.eval()
        generated = [int(t) for t in model.generate(prompt, max_tokens=128, temperature=0.0)]
        text = bytes(prompt + generated).decode("utf-8", errors="replace")
        print(f"\n  [{mode}]")
        print(f"  {repr(text)}")

    print()
    overall = all(results[m][0][-1] < TARGET_LOSS for m in results)
    print("PASSED" if overall else "FAILED", f"— target loss < {TARGET_LOSS}")


if __name__ == "__main__":
    main()
