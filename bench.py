"""
Progressive inference speedups for the diffusion UNet.

Each section applies ONE optimization and shows the speedup relative to baseline.
Techniques transfer directly to larger models (SDXL, FLUX, etc.).

Usage:
  python bench.py                              # random weights, architecture benchmarks
  python bench.py --ckpt output/ckpt_200000.pt # with trained weights

Optimizations (applied one at a time, then combined):
  1. Fewer DDIM steps    — same model, less compute, nearly same quality
  2. torch.compile       — fuses kernels, eliminates Python dispatch overhead
  3. bfloat16            — halves memory bandwidth, enables Tensor Cores on CUDA
  4. Batching            — amortizes fixed costs over multiple images
  5. Combined            — all of the above
"""
import argparse
import time

import torch
from model import UNet
from diffusion import Diffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    default=None, help="checkpoint path (optional)")
    p.add_argument("--n",       default=1,    type=int,  help="images per batch")
    p.add_argument("--warmup",  default=3,    type=int,  help="warmup runs before timing")
    p.add_argument("--runs",    default=5,    type=int,  help="timed runs to average")
    p.add_argument("--base_ch", default=128,  type=int)
    return p.parse_args()


def sync():
    """Sync device before measuring wall time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, warmup, runs):
    for _ in range(warmup): fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(runs):   fn()
    sync()
    return (time.perf_counter() - t0) / runs


def header(title):
    print(f"\n{'─'*62}\n  {title}\n{'─'*62}")


def row(label, t, baseline=None):
    speedup = f"  {baseline/t:4.1f}x faster" if baseline else ""
    print(f"  {label:<45} {t:.3f}s{speedup}")


def load_model(base_ch, ckpt_path, device):
    model = UNet(base_ch=base_ch).to(device).eval()
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    return model


def main():
    args   = parse_args()
    device = ("cuda" if torch.cuda.is_available()       else
              "mps"  if torch.backends.mps.is_available() else "cpu")

    print(f"Device:    {device}")
    print(f"Batch:     {args.n} image(s)")
    print(f"Ckpt:      {args.ckpt or 'none (random weights)'}")

    model     = load_model(args.base_ch, args.ckpt, device)
    diffusion = Diffusion(device=device)
    n         = args.n

    # ── 0. Baseline: 100 steps, float32, no compile ───────────────────────
    header("0. Baseline  (100 DDIM steps · float32 · no compile)")
    baseline = bench(lambda: diffusion.sample(model, n, steps=100), args.warmup, args.runs)
    row("100 steps, float32", baseline)

    # ── 1. Fewer DDIM steps ───────────────────────────────────────────────
    header("1. Fewer DDIM steps  (same model, proportionally less compute)")
    for steps in [50, 25, 10]:
        t = bench(lambda s=steps: diffusion.sample(model, n, steps=s), args.warmup, args.runs)
        row(f"{steps:3d} steps, float32", t, baseline)

    # ── 2. torch.compile ─────────────────────────────────────────────────
    header("2. torch.compile  (fuses ops, eliminates Python dispatch overhead)")
    compiled = torch.compile(model)
    print("  Compiling (first call is slow) ...", end="", flush=True)
    diffusion.sample(compiled, n, steps=5)   # trigger compilation
    print(" done")
    t = bench(lambda: diffusion.sample(compiled, n, steps=50), args.warmup, args.runs)
    row("50 steps, compiled", t, baseline)

    # ── 3. bfloat16 ──────────────────────────────────────────────────────
    if device in ("cuda", "mps"):
        header("3. bfloat16  (half bandwidth · Tensor Cores on CUDA · same range as float32)")
        bf_model = load_model(args.base_ch, args.ckpt, device).to(torch.bfloat16)
        # Also cast diffusion schedule tensors so arithmetic stays in bf16
        bf_diff  = Diffusion(device=device)
        for attr in ("ac", "sqrt_ac", "sqrt_one_ac"):
            setattr(bf_diff, attr, getattr(bf_diff, attr).to(torch.bfloat16))

        t = bench(lambda: bf_diff.sample(bf_model, n, steps=50), args.warmup, args.runs)
        row("50 steps, bfloat16", t, baseline)
    else:
        print("\n  (skipping bfloat16 — not on CUDA/MPS)")

    # ── 4. Larger batch ───────────────────────────────────────────────────
    if n == 1:
        header("4. Batch of 4  (amortizes attention + norm fixed costs)")
        t = bench(lambda: diffusion.sample(model, 4, steps=50), args.warmup, args.runs)
        row("50 steps, batch=4, per-image", t / 4, baseline)

    # ── 5. Combined: 25 steps + compile + bfloat16 ────────────────────────
    if device in ("cuda", "mps"):
        header("5. Combined  (25 steps · compile · bfloat16)")
        bf_compiled = torch.compile(bf_model)
        print("  Compiling ...", end="", flush=True)
        bf_diff.sample(bf_compiled, n, steps=5)
        print(" done")
        t = bench(lambda: bf_diff.sample(bf_compiled, n, steps=25), args.warmup, args.runs)
        row("25 steps, compiled, bfloat16", t, baseline)

    print(f"\n{'─'*62}")
    print(f"  Baseline: {baseline:.3f}s for {n} image(s) at 100 steps")
    print(f"{'─'*62}\n")


if __name__ == "__main__":
    main()
