"""Generate a grid of images from a trained checkpoint."""
import argparse
import torch
from torchvision import utils
from model import UNet
from diffusion import Diffusion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/ckpt_0200000.pt")
    p.add_argument("--n", type=int, default=16, help="number of images")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out", default="generated.png")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = UNet(base_ch=64).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = Diffusion(T=1000, device=device)
    samples = diffusion.sample(model, n=args.n, steps=args.steps)
    samples = (samples + 1) / 2  # [-1,1] -> [0,1]

    nrow = int(args.n ** 0.5)
    utils.save_image(samples, args.out, nrow=nrow)
    print(f"Saved {args.n} images to {args.out}")


if __name__ == "__main__":
    main()
