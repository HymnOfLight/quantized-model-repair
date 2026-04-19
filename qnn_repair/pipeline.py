"""High-level driver for the closed-loop architecture in the figure.

``RepairPipeline`` performs the following steps for a single layer (or
iteratively over several layers, mirroring the bottom feedback arrow of
the diagram):

    1. Run the model on a calibration batch and collect statistics.
    2. Generate adversarial samples (FGSM / PGD / CW).
    3. Score every neuron with the three importance estimators.
    4. Fuse the scores with weights (alpha, beta, gamma).
    5. Pick the top-k *critical* neurons.
    6. Build a :class:`RepairProblem` and solve it with the SDP or
       Gurobi backend.
    7. Write the repaired weights back into the model and report
       clean / adversarial accuracy before and after repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacks import fgsm, pgd, cw
from .importance import (
    GradientImportance,
    HessianImportance,
    ImportanceFusion,
    PerturbationImportance,
    select_critical_neurons,
)
from .quant import QuantizedMLP
from .repair import (
    GurobiRepair,
    RepairProblem,
    SDPRepair,
    apply_repair_to_model,
)


AttackFn = Callable[..., torch.Tensor]


def _attack_factory(name: str) -> AttackFn:
    name = name.lower()
    if name == "fgsm":
        return fgsm
    if name == "pgd":
        return pgd
    if name == "cw":
        return cw
    raise ValueError(f"Unknown attack: {name}")


def evaluate(
    model: nn.Module, loader: Sequence[Tuple[torch.Tensor, torch.Tensor]]
) -> float:
    device = next(model.parameters()).device
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


@dataclass
class RepairReport:
    layer_idx: int
    clean_acc_before: float
    adv_acc_before: float
    clean_acc_after: float
    adv_acc_after: float
    critical_indices: List[int]
    backend: str
    delta_norm: float


@dataclass
class RepairPipeline:
    """Closed-loop QNN repair driver.

    Parameters
    ----------
    backend:
        ``"sdp"`` or ``"gurobi"``.
    attack:
        Name of the adversarial attack (``"fgsm"``, ``"pgd"``, ``"cw"``).
    eps:
        Perturbation budget for the attack.
    top_k_ratio:
        Fraction of neurons to mark as critical (top-k by fused score).
    fusion:
        :class:`ImportanceFusion` instance.
    margin:
        Logit margin used by the Gurobi backend's hard constraints.
    lipschitz_bound:
        Optional Lipschitz constant enforced by the SDP backend.
    """

    backend: str = "sdp"
    attack: str = "pgd"
    eps: float = 0.1
    top_k_ratio: float = 0.2
    fusion: ImportanceFusion = field(default_factory=ImportanceFusion)
    margin: float = 0.5
    margin_slack_penalty: float = 10.0
    lipschitz_bound: Optional[float] = None
    delta_inf_bound: Optional[float] = None
    reg_lambda: float = 1.0
    gurobi_integer: bool = True
    gurobi_time_limit: float = 60.0
    sdp_solver: Optional[str] = None
    verbose: bool = False

    # ------------------------------------------------------------------
    def _make_attack(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        fn = _attack_factory(self.attack)
        if self.attack.lower() == "cw":
            return fn(model, x, y)
        return fn(model, x, y, eps=self.eps)

    # ------------------------------------------------------------------
    def repair_layer(
        self,
        model: QuantizedMLP,
        layer_idx: int,
        loader: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    ) -> RepairReport:
        device = next(model.parameters()).device

        # Step 1+2: gather a calibration batch and craft adversarial copies.
        x_list, y_list, x_adv_list = [], [], []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x_adv = self._make_attack(model, x, y)
            x_list.append(x)
            y_list.append(y)
            x_adv_list.append(x_adv)
        x_clean = torch.cat(x_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
        x_adv = torch.cat(x_adv_list, dim=0)

        clean_acc_before = evaluate(model, [(x_clean, y_all)])
        adv_acc_before = evaluate(model, [(x_adv, y_all)])
        if self.verbose:
            print(
                f"[layer {layer_idx}] before repair: clean={clean_acc_before:.4f} "
                f"adv={adv_acc_before:.4f}"
            )

        # Step 3+4: score and fuse.
        small_loader = [(x_clean, y_all)]
        g = GradientImportance().score(model, layer_idx, small_loader)
        h = HessianImportance(num_samples=2).score(model, layer_idx, small_loader)
        p = PerturbationImportance("activation").score(model, layer_idx, small_loader)
        fused = self.fusion(g, h, p)

        # Step 5: pick top-k critical neurons.
        n_neurons = fused.numel()
        k = max(1, int(round(self.top_k_ratio * n_neurons)))
        critical = select_critical_neurons(fused, k=k)
        if self.verbose:
            print(f"[layer {layer_idx}] critical neurons (top {k}): {critical.tolist()}")

        # Step 6: build problem and solve.
        labels = y_all if layer_idx == model.num_layers() - 1 else None
        prob = RepairProblem.from_model(
            model,
            layer_idx,
            x_clean,
            x_adv,
            labels=labels,
            critical_indices=critical,
            margin=self.margin,
            margin_slack_penalty=self.margin_slack_penalty,
            lipschitz_bound=self.lipschitz_bound,
            delta_inf_bound=self.delta_inf_bound,
            reg_lambda=self.reg_lambda,
        )

        if self.backend == "sdp":
            solver = SDPRepair(solver=self.sdp_solver, verbose=self.verbose)
        elif self.backend == "gurobi":
            solver = GurobiRepair(
                integer=self.gurobi_integer,
                time_limit=self.gurobi_time_limit,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        W_new = solver.solve(prob)

        # Step 7: write back and re-evaluate.
        delta_norm = float(((W_new - prob.W0) ** 2).sum() ** 0.5)
        apply_repair_to_model(model, layer_idx, W_new)
        clean_acc_after = evaluate(model, [(x_clean, y_all)])
        adv_acc_after = evaluate(model, [(x_adv, y_all)])
        if self.verbose:
            print(
                f"[layer {layer_idx}] after  repair: clean={clean_acc_after:.4f} "
                f"adv={adv_acc_after:.4f}  ||Delta||_F={delta_norm:.4g}"
            )

        return RepairReport(
            layer_idx=layer_idx,
            clean_acc_before=clean_acc_before,
            adv_acc_before=adv_acc_before,
            clean_acc_after=clean_acc_after,
            adv_acc_after=adv_acc_after,
            critical_indices=critical.tolist(),
            backend=self.backend,
            delta_norm=delta_norm,
        )

    # ------------------------------------------------------------------
    def repair_all(
        self,
        model: QuantizedMLP,
        loader: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        layer_indices: Optional[Sequence[int]] = None,
    ) -> List[RepairReport]:
        """Iterate the closed-loop repair over every (selected) layer."""

        if layer_indices is None:
            layer_indices = range(model.num_layers())
        reports: list[RepairReport] = []
        for li in layer_indices:
            reports.append(self.repair_layer(model, li, loader))
        return reports
