"""
Pretraining script for the Liquid (CfC) model.

Modelled on base_train.py but stripped to single-GPU, no FP8, no DDP.
Designed for RTX 3050 Laptop (4GB VRAM) with fp16 + GradScaler.

Run from project root:
    uv run python -m scripts.liquid_pretrain

Example (fast smoke run, byte-level, no dataset needed):
    uv run python -m scripts.liquid_pretrain --num-iterations 200 --seq-len 64 --device-batch-size 1

Example (real pretraining on FineWeb-EDU):
    uv run python -m scripts.liquid_pretrain --run liquid-v1 --num-iterations 5000 --seq-len 256 --device-batch-size 1 --grad-accum 16
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import math
import time

import torch
import wandb

from nanochat.common import COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, autodetect_device_type, get_peak_flops, print_banner
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

print_banner()

# -----------------------------------------------------------------------------
# CLI
parser = argparse.ArgumentParser(description="Pretrain Liquid (CfC) model")
parser.add_argument("--run", type=str, default="dummy", help="W&B run name ('dummy' disables logging)")
parser.add_argument("--num-iterations", type=int, default=1000)
parser.add_argument("--seq-len", type=int, default=256, help="sequence length (keep ≤256 for 4GB VRAM)")
parser.add_argument("--device-batch-size", type=int, default=1)
parser.add_argument("--grad-accum", type=int, default=16, help="gradient accumulation steps")
parser.add_argument("--depth", type=int, default=4, help="number of CfC layers")
parser.add_argument("--n-embd", type=int, default=256, help="hidden dimension")
parser.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
parser.add_argument("--warmup-steps", type=int, default=50)
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = never)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoint every N steps (-1 = only at end)")
parser.add_argument("--out-dir", type=str, default="runs/liquid", help="checkpoint output directory")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Device / dtype setup (single GPU, no DDP)
device_type = autodetect_device_type()
device = torch.device(device_type)
print(f"Device: {device}  |  COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print(f"GPU: {gpu_name}  |  Peak FLOPS (BF16): {gpu_peak_flops:.2e}")

# GradScaler required for fp16; bf16/fp32 don't need it
scaler = torch.cuda.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None  # type: ignore[attr-defined]
if scaler:
    print("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# W&B
use_wandb = args.run != "dummy"
if use_wandb:
    wandb_run = wandb.init(project="liquid-nanochat", name=args.run, config=vars(args))
else:
    wandb_run = None
    print("W&B disabled (pass --run <name> to enable)")

# -----------------------------------------------------------------------------
# Tokenizer + vocab size
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Model
config = GPTConfig(
    sequence_len=args.seq_len,
    vocab_size=vocab_size,
    n_layer=args.depth,
    n_head=4,        # fixed; head_dim = n_embd // 4
    n_kv_head=4,
    n_embd=args.n_embd,
    use_liquid=True,
    mlp_ratio=1,     # slim MLP for memory efficiency
)
with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()
model.train()

nparams = sum(p.numel() for p in model.parameters())
print(f"Liquid model: {nparams:,} parameters  |  depth={args.depth}  |  n_embd={args.n_embd}")

# Compile — skip if Triton is missing (not available on Windows)
def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False

if _triton_available():
    model = torch.compile(model, dynamic=False)
    print("torch.compile: enabled")
else:
    print("torch.compile: skipped (triton not available on this platform)")

# -----------------------------------------------------------------------------
# Optimizer (simple AdamW; MuonAdamW not needed for this scale)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=(device_type == "cuda"),
)

def get_lr(step: int) -> float:
    """Cosine decay with linear warmup."""
    if step < args.warmup_steps:
        return args.lr * step / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.num_iterations - args.warmup_steps)
    return args.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))

# -----------------------------------------------------------------------------
# DataLoaders
train_loader = tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.seq_len, split="train", device=device,
)
def val_loader_factory():
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.seq_len, split="val", device=device,
    )

def evaluate_val_bpb(n_batches: int = 20) -> float:
    model.eval()
    total_loss = 0.0
    val_loader = val_loader_factory()
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = next(val_loader)[:2]
            loss = model(x, targets=y)
            total_loss += loss.item()
    model.train()
    bpb = total_loss / n_batches / math.log(2)
    return bpb

# -----------------------------------------------------------------------------
# Training loop
os.makedirs(args.out_dir, exist_ok=True)
x, y = next(train_loader)[:2]  # prefetch first batch

print(f"\n{'Step':>6}  {'Loss':>8}  {'BPB':>8}  {'ms/step':>8}  {'VRAM GB':>8}")
print("-" * 55)

step_times: list[float] = []
best_val_bpb = float("inf")

for step in range(1, args.num_iterations + 1):
    # Update LR
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    t0 = time.perf_counter()
    optimizer.zero_grad()
    accum_loss = 0.0

    for _ in range(args.grad_accum):
        loss = model(x, targets=y) / args.grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        accum_loss += loss.item()
        x, y = next(train_loader)[:2]  # prefetch next batch

    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    step_ms = (time.perf_counter() - t0) * 1000
    step_times.append(step_ms)

    vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device_type == "cuda" else 0.0
    torch.cuda.reset_peak_memory_stats() if device_type == "cuda" else None

    # Evaluate val bpb
    val_bpb = float("nan")
    if args.eval_every > 0 and step % args.eval_every == 0:
        val_bpb = evaluate_val_bpb()
        if val_bpb < best_val_bpb:
            best_val_bpb = val_bpb

    # Log
    if step % 10 == 0 or step == 1:
        avg_ms = sum(step_times[-10:]) / len(step_times[-10:])
        tok_per_sec = args.device_batch_size * args.seq_len * args.grad_accum / (avg_ms / 1000)
        bpb_str = f"{val_bpb:.4f}" if not math.isnan(val_bpb) else "      -"
        print(f"{step:>6}  {accum_loss:>8.4f}  {bpb_str:>8}  {avg_ms:>8.1f}  {vram_gb:>8.3f}")

        if wandb_run is not None:
            log = {"train/loss": accum_loss, "train/lr": lr, "perf/ms_per_step": avg_ms,
                   "perf/tok_per_sec": tok_per_sec, "sys/vram_gb": vram_gb}
            if not math.isnan(val_bpb):
                log["val/bpb"] = val_bpb
            wandb_run.log(log, step=step)

    # Save checkpoint
    if args.save_every > 0 and step % args.save_every == 0:
        ckpt_path = os.path.join(args.out_dir, f"step_{step:06d}.pt")
        raw: GPT = model._orig_mod if hasattr(model, "_orig_mod") else model  # type: ignore[assignment]
        torch.save({"model": raw.state_dict(), "config": config, "step": step}, ckpt_path)
        print(f"  Saved checkpoint → {ckpt_path}")

    gc.collect()

# Final checkpoint
ckpt_path = os.path.join(args.out_dir, "final.pt")
raw_final: GPT = model._orig_mod if hasattr(model, "_orig_mod") else model  # type: ignore[assignment]
torch.save({"model": raw_final.state_dict(), "config": config, "step": args.num_iterations}, ckpt_path)
print(f"\nFinal checkpoint → {ckpt_path}")
print(f"Best val BPB: {best_val_bpb:.4f}")

if wandb_run is not None:
    wandb_run.finish()
