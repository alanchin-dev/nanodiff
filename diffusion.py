"""
DDPM noise schedule + DDIM sampling.

Key equations:
  Forward  q(x_t | x_0) = N(√ᾱ_t · x_0,  (1-ᾱ_t)·I)       [corrupt x_0 to x_t]
  Reverse  ε_θ(x_t, t) predicts the noise added at step t     [model's job]
  DDIM     x_{t-1} = √ᾱ_{t-1}·x̂_0  +  √(1-ᾱ_{t-1})·ε_θ    [denoise, η=0 case]

DDIM (Song et al. 2020) lets you denoise in far fewer steps than T (e.g. 50 vs 1000)
by skipping timesteps while reusing the same model trained with DDPM.
"""
import math
import torch


def cosine_alphas_cumprod(T, s=0.008):
    """
    Cosine noise schedule (Nichol & Dhariwal 2021, "Improved DDPM").
    Returns ᾱ_1 ... ᾱ_T, the cumulative product of (1 - β_t).
    Smoother than linear: avoids destroying signal too fast at the start.
    """
    steps = torch.arange(T + 1)
    f  = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]                    # normalize to ᾱ_0 = 1
    return ac[1:].clamp(0, 1)        # shape (T,): ᾱ_1 ... ᾱ_T


class Diffusion:
    """
    Forward process (training): adds noise to images.
    Reverse process (inference): denoises via DDIM with configurable step count.
    """
    def __init__(self, T=1000, device="cpu"):
        self.T  = T
        ac                 = cosine_alphas_cumprod(T).to(device)
        self.ac            = ac                   # ᾱ_t
        self.sqrt_ac       = ac.sqrt()            # √ᾱ_t
        self.sqrt_one_ac   = (1 - ac).sqrt()      # √(1-ᾱ_t)

    def q_sample(self, x0, t, noise=None):
        """Forward: corrupt x0 to x_t by mixing with Gaussian noise."""
        noise = noise if noise is not None else torch.randn_like(x0)
        s = self.sqrt_ac[t][:, None, None, None]
        r = self.sqrt_one_ac[t][:, None, None, None]
        return s * x0 + r * noise, noise

    @torch.no_grad()
    def sample(self, model, n, steps=50, img_size=(3, 32, 32), eta=0.0):
        """
        DDIM reverse process: start from Gaussian noise, iteratively denoise.

        steps: how many denoising steps to take (50 ≈ same quality as 1000-step DDPM)
        eta:   0.0 = deterministic DDIM,  1.0 = stochastic DDPM-like
        """
        dev = self.ac.device
        x   = torch.randn(n, *img_size, device=dev)

        # Uniformly spaced timesteps T-1 → 0, e.g. [999, 978, 958, ..., 0] for steps=50
        ts = torch.linspace(self.T - 1, 0, steps + 1).long()

        for i in range(steps):
            t_cur  = ts[i].item()
            t_prev = ts[i + 1].item()
            t_batch = torch.full((n,), t_cur, dtype=torch.long, device=dev)

            # Predict noise, then estimate clean image
            eps    = model(x, t_batch)
            x0_hat = (x - self.sqrt_one_ac[t_cur] * eps) / self.sqrt_ac[t_cur]
            x0_hat = x0_hat.clamp(-1, 1)

            ac_cur  = self.ac[t_cur]
            ac_prev = self.ac[t_prev]

            if eta == 0.0:
                # Deterministic DDIM — no randomness, fastest and most reproducible
                x = ac_prev.sqrt() * x0_hat + (1 - ac_prev).sqrt() * eps
            else:
                # Stochastic DDIM (eta=1 recovers DDPM variance)
                sigma = eta * ((1 - ac_prev) / (1 - ac_cur) * (1 - ac_cur / ac_prev)).sqrt()
                x  = ac_prev.sqrt() * x0_hat
                x += (1 - ac_prev - sigma**2).clamp(min=0).sqrt() * eps
                x += sigma * torch.randn_like(x)

        return x.clamp(-1, 1)
