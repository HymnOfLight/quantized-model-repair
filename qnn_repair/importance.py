"""Neuron-importance estimators (middle panel of the architecture).

Three families of estimators are implemented, mirroring the figure:

1. **Gradient based**

   * 1st-order Taylor expansion::

         I_i = sum_x | w_i * dL/dw_i |

   * Integrated gradients::

         I_i = (1/L2) * sqrt( sum_x ( w_i * dL/dw_i )^2 )

2. **Hessian based**

   * Trace estimate via Hutchinson's stochastic trace estimator.
   * Top-k eigenvalue spectrum approximation via subspace iteration.

3. **Perturbation / output based**

   * Activation magnitude statistics (mean abs activation).
   * Output sensitivity: drop a neuron, measure ``Delta L`` (occlusion).

The :class:`ImportanceFusion` module fuses the three scores with learnable
weights ``alpha, beta, gamma`` (the ``α/β/γ`` symbol in the figure) and the
helper :func:`select_critical_neurons` returns the indices of the ``k`` most
important neurons in a chosen layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import QuantizedLinear, QuantizedMLP


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _layer_loss(
    model: QuantizedMLP, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    return F.cross_entropy(model(x), y)


def _normalise(scores: torch.Tensor) -> torch.Tensor:
    s = scores.detach()
    if s.numel() == 0:
        return s
    smax = s.abs().max().clamp(min=1e-12)
    return s / smax


# ---------------------------------------------------------------------------
# 1st-order gradient based importance
# ---------------------------------------------------------------------------


class GradientImportance:
    """First-order Taylor / integrated-gradient importance per neuron.

    Importance is summed over the rows of the weight matrix of a chosen
    hidden layer, giving one score per *output* neuron of that layer.
    """

    def __init__(self, mode: str = "taylor"):
        if mode not in {"taylor", "integrated"}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode

    def score(
        self,
        model: QuantizedMLP,
        layer_idx: int,
        loader: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        layer: QuantizedLinear = model.hidden_layer(layer_idx)
        w = layer.linear.weight
        sum_abs = torch.zeros(w.size(0), device=w.device)
        sum_sq = torch.zeros(w.size(0), device=w.device)
        n = 0
        for x, y in loader:
            x = x.to(w.device)
            y = y.to(w.device)
            model.zero_grad(set_to_none=True)
            loss = _layer_loss(model, x, y)
            grad = torch.autograd.grad(loss, w, retain_graph=False)[0]
            contrib = (w * grad).abs()
            sum_abs += contrib.sum(dim=1)
            sum_sq += (w * grad).pow(2).sum(dim=1)
            n += x.size(0)
        if self.mode == "taylor":
            return _normalise(sum_abs / max(n, 1))
        return _normalise(torch.sqrt(sum_sq) / max(n, 1))


# ---------------------------------------------------------------------------
# Hessian-trace importance (Hutchinson estimator)
# ---------------------------------------------------------------------------


class HessianImportance:
    """Diagonal Hessian-trace importance per neuron via Hutchinson.

    For each Rademacher random vector ``v`` we compute ``g = dL/dw`` then
    ``Hv = d(g . v)/dw``.  ``Hv ⊙ v`` is an unbiased estimator of
    ``diag(H)``.  Summing over the columns of a layer gives one score per
    output neuron.
    """

    def __init__(self, num_samples: int = 4):
        self.num_samples = num_samples

    def score(
        self,
        model: QuantizedMLP,
        layer_idx: int,
        loader: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        layer = model.hidden_layer(layer_idx)
        w = layer.linear.weight
        diag = torch.zeros_like(w)
        n_batches = 0
        for x, y in loader:
            x = x.to(w.device)
            y = y.to(w.device)
            for _ in range(self.num_samples):
                model.zero_grad(set_to_none=True)
                loss = _layer_loss(model, x, y)
                grad = torch.autograd.grad(loss, w, create_graph=True)[0]
                v = torch.randint_like(w, low=0, high=2, dtype=w.dtype).mul_(2).sub_(1)
                hv = torch.autograd.grad((grad * v).sum(), w, retain_graph=False)[0]
                diag = diag + (hv * v).detach()
            n_batches += 1
        diag = diag / max(self.num_samples * n_batches, 1)
        return _normalise(diag.abs().sum(dim=1))


# ---------------------------------------------------------------------------
# Perturbation / activation importance
# ---------------------------------------------------------------------------


class PerturbationImportance:
    """Activation magnitude + occlusion based importance.

    ``mode='activation'`` returns the mean absolute post-ReLU activation
    of every neuron.  ``mode='occlusion'`` zeros each neuron in turn and
    measures the increase in cross-entropy loss.
    """

    def __init__(self, mode: str = "activation", max_neurons_per_pass: int = 64):
        if mode not in {"activation", "occlusion"}:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        self.max_neurons_per_pass = max_neurons_per_pass

    def _activation_score(
        self,
        model: QuantizedMLP,
        layer_idx: int,
        loader: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        n = 0
        accum: Optional[torch.Tensor] = None
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits, acts = model.forward_with_activations(x)
                # The output layer is not in ``acts`` (it has no ReLU);
                # use its raw pre-activation logits in that case.
                a_tensor = logits if layer_idx >= len(acts) else acts[layer_idx]
                a = a_tensor.abs().mean(dim=0)
                accum = a if accum is None else accum + a
                n += 1
        assert accum is not None
        return _normalise(accum / max(n, 1))

    def _occlusion_score(
        self,
        model: QuantizedMLP,
        layer_idx: int,
        loader: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        layer = model.hidden_layer(layer_idx)
        n_neurons = layer.linear.weight.size(0)
        scores = torch.zeros(n_neurons, device=device)

        baseline_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for x, y in loader:
                baseline_loss += _layer_loss(model, x.to(device), y.to(device)).item()
                n_batches += 1
        baseline_loss /= max(n_batches, 1)

        original_weight = layer.linear.weight.data.clone()
        original_bias = (
            layer.linear.bias.data.clone() if layer.linear.bias is not None else None
        )
        with torch.no_grad():
            for i in range(n_neurons):
                layer.linear.weight.data[i].zero_()
                if layer.linear.bias is not None:
                    layer.linear.bias.data[i] = 0.0

                loss = 0.0
                for x, y in loader:
                    loss += _layer_loss(model, x.to(device), y.to(device)).item()
                loss /= max(n_batches, 1)
                scores[i] = loss - baseline_loss

                layer.linear.weight.data[i].copy_(original_weight[i])
                if layer.linear.bias is not None and original_bias is not None:
                    layer.linear.bias.data[i] = original_bias[i]
        return _normalise(scores.abs())

    def score(
        self,
        model: QuantizedMLP,
        layer_idx: int,
        loader: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if self.mode == "activation":
            return self._activation_score(model, layer_idx, loader)
        return self._occlusion_score(model, layer_idx, loader)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


@dataclass
class ImportanceFusion:
    """Convex combination of the three importance families.

    The weights are renormalised so they sum to 1.  Defaults give equal
    weight to each family (``α = β = γ = 1/3``).
    """

    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0

    def __call__(
        self,
        gradient_score: torch.Tensor,
        hessian_score: torch.Tensor,
        perturbation_score: torch.Tensor,
    ) -> torch.Tensor:
        s = self.alpha + self.beta + self.gamma
        if s <= 0:
            raise ValueError("alpha+beta+gamma must be > 0")
        a, b, g = self.alpha / s, self.beta / s, self.gamma / s
        return a * gradient_score + b * hessian_score + g * perturbation_score


def select_critical_neurons(
    fused_scores: torch.Tensor, k: Optional[int] = None, threshold: Optional[float] = None
) -> torch.Tensor:
    """Return the indices of the most important neurons.

    Either ``k`` (top-k) or ``threshold`` (score >= threshold) must be
    provided.  ``k`` takes precedence when both are given.
    """

    if k is not None:
        k = max(0, min(k, fused_scores.numel()))
        if k == 0:
            return torch.empty(0, dtype=torch.long, device=fused_scores.device)
        return torch.topk(fused_scores, k=k).indices.sort().values
    if threshold is not None:
        return torch.nonzero(fused_scores >= threshold, as_tuple=False).flatten()
    raise ValueError("Either k or threshold must be specified")
