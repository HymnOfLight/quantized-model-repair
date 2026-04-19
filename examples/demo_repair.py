"""End-to-end demonstration of the QNN-repair pipeline.

The script trains a small mixed-precision MLP on a synthetic two-moons-
like dataset, evaluates it under a PGD attack, identifies the critical
neurons of the first hidden layer, and finally calls the SDP backend
(or Gurobi if available) to repair the layer.

Run::

    python examples/demo_repair.py --backend sdp
    python examples/demo_repair.py --backend gurobi
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from qnn_repair import (
    ImportanceFusion,
    QuantizedMLP,
    RepairPipeline,
)


def make_dataset(n: int = 1024, seed: int = 0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, np.pi, size=n)
    r = 1.0
    x0 = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    x1 = np.stack([r * np.cos(theta) + 1.0, -r * np.sin(theta) + 0.5], axis=1)
    x = np.concatenate([x0, x1], axis=0).astype(np.float32)
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])
    x += 0.1 * rng.standard_normal(x.shape).astype(np.float32)
    # Scale to [0, 1] so the FGSM/PGD clip range is meaningful.
    x = (x - x.min(0)) / (x.max(0) - x.min(0) + 1e-8)
    perm = rng.permutation(len(x))
    return torch.from_numpy(x[perm]), torch.from_numpy(y[perm])


def train(model: QuantizedMLP, loader, epochs: int = 30, lr: float = 1e-2) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sdp", "gurobi"], default="sdp")
    parser.add_argument("--attack", choices=["fgsm", "pgd", "cw"], default="pgd")
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--top-k", type=float, default=0.4)
    parser.add_argument("--lipschitz", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Also repair the output layer (default: only hidden layers)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y = make_dataset()
    train_ds = TensorDataset(x[:1500], y[:1500])
    cal_ds = TensorDataset(x[1500:], y[1500:])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    cal_loader = DataLoader(cal_ds, batch_size=128, shuffle=False)

    model = QuantizedMLP(
        in_features=2,
        hidden=[16, 8],
        out_features=2,
        bits=[8, 4, 2],
        act_bits=8,
    )

    print("Training quantised MLP ...")
    train(model, train_loader)

    pipeline = RepairPipeline(
        backend=args.backend,
        attack=args.attack,
        eps=args.eps,
        top_k_ratio=args.top_k,
        fusion=ImportanceFusion(alpha=1.0, beta=0.5, gamma=0.5),
        lipschitz_bound=args.lipschitz,
        verbose=True,
    )

    layer_indices = None if args.all_layers else range(model.num_hidden_layers())
    reports = pipeline.repair_all(model, cal_loader, layer_indices=layer_indices)
    print()
    print("=== Repair summary ===")
    for r in reports:
        print(
            f"layer {r.layer_idx} [{r.backend}]: "
            f"clean {r.clean_acc_before:.3f} -> {r.clean_acc_after:.3f},  "
            f"adv {r.adv_acc_before:.3f} -> {r.adv_acc_after:.3f},  "
            f"||Delta||_F={r.delta_norm:.4g}"
        )


if __name__ == "__main__":
    main()
