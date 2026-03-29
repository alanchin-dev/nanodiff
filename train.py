"""
Train the UNet on CIFAR-10 with the DDPM objective.

Loss: MSE between predicted noise and actual noise added at each timestep.
Quality becomes recognizable around 50k steps; gets good around 200k steps.

Usage:
  python train.py                        # default settings
  python train.py --base_ch 64           # smaller model, faster to train
  python train.py --resume output/ckpt_0050000.pt
"""
import argparse
import os
import ssl

import certifi
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import UNet
from diffusion import Diffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_ch",    type=int,   default=128)
    p.add_argument("--T",          type=int,   default=1000)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--n_steps",    type=int,   default=200_000)
    p.add_argument("--log_every",  type=int,   default=1_000)
    p.add_argument("--save_every", type=int,   default=10_000)
    p.add_argument("--out_dir",    type=str,   default="output")
    p.add_argument("--resume",     type=str,   default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = ("cuda"  if torch.cuda.is_available()  else
              "mps"   if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Fix SSL cert verification inside a venv on macOS: point the default HTTPS
    # context at certifi's CA bundle instead of disabling verification entirely.
    # This preserves full certificate validation while working without system certs.
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

    # CIFAR-10, normalized to [-1, 1]
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    # MPS (Apple Silicon) doesn't support multiprocess workers or pin_memory.
    # CUDA benefits from both; CPU is fine with workers but not pin_memory.
    use_workers  = 4 if device == "cuda" else 0
    use_pin_mem  = device == "cuda"
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=use_workers, pin_memory=use_pin_mem, drop_last=True)

    model     = UNet(base_ch=args.base_ch).to(device)
    diffusion = Diffusion(T=args.T, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    data_iter = iter(loader)
    loss_ema  = None

    for step in range(start_step + 1, args.n_steps + 1):
        try:
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x0, _    = next(data_iter)

        x0 = x0.to(device)
        t  = torch.randint(0, args.T, (x0.shape[0],), device=device)

        xt, noise = diffusion.q_sample(x0, t)
        pred      = model(xt, t)
        loss      = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_ema = loss.item() if loss_ema is None else 0.99 * loss_ema + 0.01 * loss.item()

        if step % args.log_every == 0:
            print(f"step {step:7d} | loss {loss_ema:.4f}")

        if step % args.save_every == 0:
            path = f"{args.out_dir}/ckpt_{step:07d}.pt"
            torch.save({"step": step, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, path)

            model.eval()
            samples = diffusion.sample(model, n=16, steps=50)
            samples = (samples + 1) / 2                    # [-1,1] → [0,1]
            utils.save_image(samples, f"{args.out_dir}/samples_{step:07d}.png", nrow=4)
            model.train()
            print(f"  → saved {path}")


if __name__ == "__main__":
    main()
