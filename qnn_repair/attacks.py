"""Adversarial attack generators used as the *adversarial-sample* branch
in the input panel of the architecture diagram.

All attacks share the same signature::

    x_adv = attack(model, x, y, eps=..., **kwargs)

where ``x`` is a clean batch and ``y`` are the integer class labels.
The returned tensor lives in the same valid range ``[x_min, x_max]`` as the
input.  Gradients of ``model`` are temporarily enabled even if the model is
in eval mode.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _project(x: torch.Tensor, x_min: float, x_max: float) -> torch.Tensor:
    return x.clamp(min=x_min, max=x_max)


def fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.1,
    clip: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """One-step Fast Gradient Sign Method attack."""

    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv.detach() + eps * grad.sign()
    return _project(x_adv, *clip)


def pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 0.1,
    alpha: float = 0.02,
    steps: int = 20,
    random_start: bool = True,
    clip: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Projected Gradient Descent (L_inf) attack."""

    x_adv = x.detach().clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = _project(x_adv, *clip)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = _project(x_adv, *clip)
    return x_adv.detach()


def cw(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 100,
    lr: float = 0.01,
    clip: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """Carlini & Wagner L_2 attack (un-targeted variant)."""

    x_min, x_max = clip
    # Map x into (-inf, inf) via inverse-tanh so the constraint is implicit.
    x_scaled = (x - x_min) / (x_max - x_min)
    x_scaled = x_scaled.clamp(1e-6, 1 - 1e-6)
    w = torch.atanh(2 * x_scaled - 1).detach().clone().requires_grad_(True)
    optim = torch.optim.Adam([w], lr=lr)

    for _ in range(steps):
        x_adv = (torch.tanh(w) + 1) / 2 * (x_max - x_min) + x_min
        logits = model(x_adv)
        one_hot = F.one_hot(y, num_classes=logits.size(1)).bool()
        real = logits.masked_select(one_hot)
        other = logits.masked_fill(one_hot, float("-inf")).max(dim=1).values
        f = torch.clamp(real - other + kappa, min=0)
        l2 = (x_adv - x).flatten(1).pow(2).sum(dim=1)
        loss = (l2 + c * f).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        x_adv = (torch.tanh(w) + 1) / 2 * (x_max - x_min) + x_min
    return x_adv.detach()
