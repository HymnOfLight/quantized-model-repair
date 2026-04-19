"""Quantised neural network primitives.

This module implements the *left* panel of the architecture diagram:

* :class:`QuantConfig`   -- per-tensor quantisation configuration
  (bit-width, signed/unsigned, symmetric/asymmetric).
* :func:`quantize_tensor` and :class:`FakeQuantize` -- simulated-quantisation
  with a straight-through estimator (STE) so that gradients flow through the
  quantiser during training and importance-scoring.
* :class:`QuantizedMLP` -- a small but fully featured MLP that uses
  :class:`FakeQuantize` on every weight and activation.  Mixed-precision
  configurations such as the ``INT8 / INT4 / INT2`` mix shown in the figure
  are supported by passing a list of bit-widths.

The implementation purposefully uses *fake* (simulated) quantisation rather
than real INT-only kernels so the same model can be used both for
adversarial-attack generation and for the SDP / Gurobi based repair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quant config + fake quantiser
# ---------------------------------------------------------------------------


@dataclass
class QuantConfig:
    """Per-tensor quantisation configuration.

    Parameters
    ----------
    bits:
        Number of bits used to represent the quantised value.
    signed:
        Whether the quantised range is signed.
    symmetric:
        If ``True`` use a symmetric range ``[-2**(b-1)+1, 2**(b-1)-1]`` so
        the zero point is exactly representable.  Asymmetric quantisation
        uses an offset/zero-point.
    per_channel:
        If ``True`` the scale/zero-point are computed per output channel
        rather than per tensor.  Only used for weight tensors.
    """

    bits: int = 8
    signed: bool = True
    symmetric: bool = True
    per_channel: bool = False

    def qmin_qmax(self) -> tuple[int, int]:
        if self.signed:
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -qmax if self.symmetric else -(2 ** (self.bits - 1))
        else:
            qmin = 0
            qmax = 2**self.bits - 1
        return qmin, qmax


def _compute_scale_zp(
    x: torch.Tensor, cfg: QuantConfig, dim: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    qmin, qmax = cfg.qmin_qmax()
    if dim is None:
        x_min = x.min().detach()
        x_max = x.max().detach()
    else:
        reduce_dims = [d for d in range(x.ndim) if d != dim]
        x_min = x.amin(dim=reduce_dims, keepdim=False).detach()
        x_max = x.amax(dim=reduce_dims, keepdim=False).detach()

    if cfg.symmetric:
        bound = torch.maximum(x_max.abs(), x_min.abs()).clamp(min=1e-8)
        scale = bound / qmax
        zero_point = torch.zeros_like(scale)
    else:
        scale = (x_max - x_min).clamp(min=1e-8) / (qmax - qmin)
        zero_point = qmin - torch.round(x_min / scale)
    return scale, zero_point


def quantize_tensor(
    x: torch.Tensor, cfg: QuantConfig, dim: Optional[int] = None
) -> torch.Tensor:
    """Fake-quantise ``x`` according to ``cfg`` with a straight-through
    estimator.  The output has the same dtype and shape as ``x``.
    """

    qmin, qmax = cfg.qmin_qmax()
    scale, zero_point = _compute_scale_zp(x, cfg, dim=dim)
    if dim is not None:
        view_shape = [1] * x.ndim
        view_shape[dim] = -1
        scale = scale.view(view_shape)
        zero_point = zero_point.view(view_shape)
    q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    x_dq = (q - zero_point) * scale
    # Straight-through estimator: forward = quantised, backward = identity.
    return x + (x_dq - x).detach()


class FakeQuantize(nn.Module):
    """:class:`torch.nn.Module` wrapper around :func:`quantize_tensor`."""

    def __init__(self, cfg: QuantConfig, dim: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return quantize_tensor(x, self.cfg, dim=self.dim)

    def extra_repr(self) -> str:
        return (
            f"bits={self.cfg.bits}, signed={self.cfg.signed}, "
            f"symmetric={self.cfg.symmetric}, per_channel={self.cfg.per_channel}"
        )


# ---------------------------------------------------------------------------
# Quantised MLP
# ---------------------------------------------------------------------------


class QuantizedLinear(nn.Module):
    """Linear layer whose weights and activations are fake-quantised."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_cfg: QuantConfig,
        act_cfg: Optional[QuantConfig] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.w_quant = FakeQuantize(
            weight_cfg, dim=0 if weight_cfg.per_channel else None
        )
        self.a_quant = FakeQuantize(act_cfg) if act_cfg is not None else None

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w_quant(self.linear.weight)
        out = F.linear(x, w, self.linear.bias)
        if self.a_quant is not None:
            out = self.a_quant(out)
        return out


class QuantizedMLP(nn.Module):
    """Multi-layer perceptron with mixed-precision fake quantisation.

    The figure shows a network with INT8 / INT4 / INT2 layers; pass a list of
    ``bits`` of length ``len(hidden)+1`` to reproduce that mix.  Discrete
    activations are obtained by quantising the post-ReLU outputs.
    """

    def __init__(
        self,
        in_features: int,
        hidden: Sequence[int],
        out_features: int,
        bits: Sequence[int] | int = 8,
        act_bits: Optional[int] = 8,
        per_channel_weights: bool = False,
    ):
        super().__init__()
        sizes = [in_features, *hidden, out_features]
        if isinstance(bits, int):
            bits = [bits] * (len(sizes) - 1)
        if len(bits) != len(sizes) - 1:
            raise ValueError("len(bits) must equal number of layers")

        self.layers: nn.ModuleList = nn.ModuleList()
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            w_cfg = QuantConfig(bits=bits[i], per_channel=per_channel_weights)
            a_cfg = (
                QuantConfig(bits=act_bits, signed=False, symmetric=False)
                if act_bits is not None and i < len(sizes) - 2
                else None
            )
            self.layers.append(QuantizedLinear(a, b, w_cfg, a_cfg))
        self.in_features = in_features
        self.out_features = out_features

    # ------------------------------------------------------------------
    # Forward + introspection helpers
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def forward_with_activations(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """Run a forward pass and return the post-activation outputs of every
        hidden layer (used by importance estimators)."""

        acts: List[torch.Tensor] = []
        x = x.flatten(1)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                acts.append(x)
        return x, acts

    # ------------------------------------------------------------------
    # Repair helpers
    # ------------------------------------------------------------------
    def hidden_layer(self, idx: int) -> QuantizedLinear:
        """Return the ``idx``-th hidden :class:`QuantizedLinear`."""
        return self.layers[idx]

    def num_hidden_layers(self) -> int:
        """Number of layers excluding the output layer."""
        return len(self.layers) - 1

    def num_layers(self) -> int:
        """Total number of linear layers (including the output layer)."""
        return len(self.layers)

    def disable_quant(self) -> None:
        """Disable fake-quantisation (handy when computing Hessians)."""
        for m in self.modules():
            if isinstance(m, FakeQuantize):
                m.enabled = False

    def enable_quant(self) -> None:
        for m in self.modules():
            if isinstance(m, FakeQuantize):
                m.enabled = True
