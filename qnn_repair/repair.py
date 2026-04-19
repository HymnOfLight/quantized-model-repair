r"""Quantised neural-network repair backends.

The figure shows the importance-evaluation panel feeding a ``Critical
Neuron Set`` :math:`\mathcal{S}_{\text{crit}}` into the rightmost panel,
which then drives "model pruning / mixed precision / robustness
enhancement / interpretability" applications.  This module implements the
*robustness enhancement* arrow concretely: given a critical neuron set,
repair the layer's weights so the network behaves correctly on a
calibration set (clean **and** adversarial) while keeping the change as
small as possible and preserving quantisability.

Two interchangeable backends are provided:

* :class:`GurobiRepair` -- mixed-integer quadratic programming (MIQP).
  Weight perturbations are forced onto the discrete quantisation grid via
  integer variables; correctness constraints are added as hard linear
  margin constraints; the squared-error reconstruction loss is the
  quadratic objective.

* :class:`SDPRepair` -- convex semidefinite programming relaxation
  formulated in CVXPY.  Weight perturbations are continuous; the layer's
  Lipschitz constant is upper-bounded via a Schur-complement LMI which
  controls the post-repair worst-case sensitivity to input perturbations.

Both backends consume the common :class:`RepairProblem` description so the
pipeline can call either.

The maths
---------

Let :math:`W_0 \in \mathbb{R}^{n_o \times n_i}` and :math:`b_0` be the
original weights of the target layer; :math:`A \in \mathbb{R}^{N \times n_i}`
the activations entering the layer for ``N`` calibration samples; and
:math:`Y \in \mathbb{R}^{N \times n_o}` the *desired* pre-activation
outputs.  Let :math:`\mathcal{S}` be the critical-neuron mask
(`1` for rows that may change, `0` otherwise).  We solve

.. math::
    \min_{\Delta} \; \|\Delta\|_F^2 + \lambda
        \|A (W_0 + \Delta)^\top + \mathbf{1} b_0^\top - Y\|_F^2

subject to

* :math:`\Delta_{j,:} = 0` for :math:`j \notin \mathcal{S}`
  (only critical rows may move),
* (Gurobi) :math:`W_0 + \Delta` lies on the quantisation grid,
* (Gurobi, optional) classification margin constraints
  :math:`(W+\Delta)_{y_i,:} a_i - (W+\Delta)_{j,:} a_i \ge m`,
* (SDP, optional) Lipschitz bound :math:`\sigma_{\max}(W_0+\Delta) \le L`
  encoded by the LMI

  .. math::
     \begin{bmatrix} L\, I & (W_0+\Delta)^\top \\
                     W_0+\Delta & L\, I \end{bmatrix} \succeq 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch

from .quant import QuantConfig, QuantizedLinear, QuantizedMLP


# ---------------------------------------------------------------------------
# Problem description
# ---------------------------------------------------------------------------


@dataclass
class RepairProblem:
    """Algebraic description of a single-layer repair problem.

    Parameters
    ----------
    W0, b0:
        Original weights and bias of the target layer.
    A:
        Activations entering the layer, shape ``(N, n_in)``.
    Y_target:
        Desired *pre-activation* output, shape ``(N, n_out)``.
    critical_indices:
        Indices of the rows of ``W0`` that may be modified.  Other rows
        are frozen.
    weight_cfg:
        Quantisation configuration of the layer (used by the Gurobi
        backend to enforce the discrete grid).
    labels:
        Optional integer class labels for each sample.  When provided,
        the Gurobi backend will add classification-margin constraints.
    margin:
        Logit margin enforced by the Gurobi backend when ``labels`` is
        given.
    delta_inf_bound:
        Optional ``L_inf`` budget on the per-weight perturbation.
    lipschitz_bound:
        Optional upper bound on ``sigma_max(W_0 + Delta)`` enforced by
        the SDP backend.
    reg_lambda:
        Weight of the reconstruction term in the objective.
    """

    W0: np.ndarray
    b0: np.ndarray
    A: np.ndarray
    Y_target: np.ndarray
    critical_indices: np.ndarray
    weight_cfg: QuantConfig = field(default_factory=QuantConfig)
    labels: Optional[np.ndarray] = None
    margin: float = 0.0
    delta_inf_bound: Optional[float] = None
    lipschitz_bound: Optional[float] = None
    reg_lambda: float = 1.0
    margin_slack_penalty: float = 10.0  # weight on slack vars; 0 = hard margin

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_model(
        cls,
        model: QuantizedMLP,
        layer_idx: int,
        x_clean: torch.Tensor,
        x_adv: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        critical_indices: torch.Tensor,
        **kwargs,
    ) -> "RepairProblem":
        """Build a :class:`RepairProblem` for ``layer_idx`` of ``model``.

        The desired pre-activation outputs for adversarial samples are
        taken from the *clean* forward pass, so the repaired layer drives
        adversarial activations back towards their clean-input target.
        """

        device = next(model.parameters()).device
        x_clean = x_clean.to(device)
        if x_adv is not None:
            x_adv = x_adv.to(device)

        with torch.no_grad():
            # Inputs to the target layer (post-ReLU activations of the
            # previous block) for both clean and adversarial samples.
            def acts_in(x: torch.Tensor) -> torch.Tensor:
                if layer_idx == 0:
                    return x.flatten(1)
                _, hidden_acts = model.forward_with_activations(x)
                return hidden_acts[layer_idx - 1]

            A_clean = acts_in(x_clean)
            target_layer: QuantizedLinear = model.hidden_layer(layer_idx)
            W = target_layer.linear.weight
            b = (
                target_layer.linear.bias
                if target_layer.linear.bias is not None
                else torch.zeros(W.size(0), device=device)
            )
            # Desired pre-activation = the clean forward through the
            # original layer.
            Y_clean = A_clean @ W.t() + b

            if x_adv is not None:
                A_adv = acts_in(x_adv)
                A_all = torch.cat([A_clean, A_adv], dim=0)
                Y_all = torch.cat([Y_clean, Y_clean], dim=0)
                if labels is not None:
                    labels_all = torch.cat([labels, labels], dim=0)
                else:
                    labels_all = None
            else:
                A_all = A_clean
                Y_all = Y_clean
                labels_all = labels

        return cls(
            W0=W.detach().cpu().numpy().astype(np.float64),
            b0=b.detach().cpu().numpy().astype(np.float64),
            A=A_all.detach().cpu().numpy().astype(np.float64),
            Y_target=Y_all.detach().cpu().numpy().astype(np.float64),
            critical_indices=critical_indices.detach().cpu().numpy().astype(np.int64),
            weight_cfg=target_layer.w_quant.cfg,
            labels=None if labels_all is None else labels_all.detach().cpu().numpy().astype(np.int64),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Helpers shared by the two backends
# ---------------------------------------------------------------------------


def _row_mask(n_out: int, critical: np.ndarray) -> np.ndarray:
    mask = np.zeros(n_out, dtype=bool)
    mask[critical] = True
    return mask


def apply_repair_to_model(
    model: QuantizedMLP, layer_idx: int, W_new: np.ndarray
) -> None:
    """Write ``W_new`` back into the target layer of ``model``."""

    layer = model.hidden_layer(layer_idx)
    with torch.no_grad():
        layer.linear.weight.copy_(
            torch.from_numpy(W_new).to(layer.linear.weight)
        )


# ---------------------------------------------------------------------------
# Gurobi backend
# ---------------------------------------------------------------------------


class GurobiRepair:
    """MIQP-based repair using Gurobi.

    Continuous formulation by default (``integer=False``) which yields a
    standard QP and runs without a Gurobi license under the size-limited
    free version.  Set ``integer=True`` to enforce the quantisation grid
    exactly using one integer variable per modifiable weight.
    """

    def __init__(
        self,
        integer: bool = True,
        time_limit: float = 60.0,
        mip_gap: float = 1e-3,
        verbose: bool = False,
    ):
        self.integer = integer
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose

    # ------------------------------------------------------------------
    def solve(self, prob: RepairProblem) -> np.ndarray:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:  # pragma: no cover - exercised on import
            raise RuntimeError(
                "GurobiRepair requires the `gurobipy` package. Install it "
                "with `pip install gurobipy` and configure a license."
            ) from exc

        W0 = prob.W0
        b0 = prob.b0
        A = prob.A
        Y = prob.Y_target
        n_out, n_in = W0.shape
        N = A.shape[0]
        crit_mask = _row_mask(n_out, prob.critical_indices)

        # Quantisation grid.
        qmin, qmax = prob.weight_cfg.qmin_qmax()
        scale = float(np.maximum(np.abs(W0).max(), 1e-8) / qmax)

        model = gp.Model("qnn_repair_milp")
        model.Params.OutputFlag = 1 if self.verbose else 0
        model.Params.TimeLimit = self.time_limit
        if self.integer:
            model.Params.MIPGap = self.mip_gap

        # Decision variables: one per (j, k).  Frozen rows are fixed.
        W_new: dict[tuple[int, int], gp.Var] = {}
        for j in range(n_out):
            for k in range(n_in):
                if not crit_mask[j]:
                    var = model.addVar(
                        lb=W0[j, k], ub=W0[j, k], name=f"w_{j}_{k}"
                    )
                else:
                    if self.integer:
                        q = model.addVar(
                            lb=qmin, ub=qmax, vtype=GRB.INTEGER, name=f"q_{j}_{k}"
                        )
                        var = model.addVar(
                            lb=qmin * scale, ub=qmax * scale, name=f"w_{j}_{k}"
                        )
                        model.addConstr(var == scale * q)
                    else:
                        lb = qmin * scale
                        ub = qmax * scale
                        if prob.delta_inf_bound is not None:
                            lb = max(lb, W0[j, k] - prob.delta_inf_bound)
                            ub = min(ub, W0[j, k] + prob.delta_inf_bound)
                        var = model.addVar(lb=lb, ub=ub, name=f"w_{j}_{k}")
                W_new[j, k] = var
        model.update()

        # Pre-activation outputs Z_{i, j} = sum_k A_{i,k} * W_{j, k} + b_j.
        Z: dict[tuple[int, int], gp.LinExpr] = {}
        for i in range(N):
            for j in range(n_out):
                Z[i, j] = (
                    gp.quicksum(A[i, k] * W_new[j, k] for k in range(n_in))
                    + b0[j]
                )

        # Objective: ||Delta||_F^2 + lambda * ||Z - Y||_F^2.
        obj = gp.QuadExpr()
        for j in range(n_out):
            for k in range(n_in):
                d = W_new[j, k] - W0[j, k]
                obj += d * d
        for i in range(N):
            for j in range(n_out):
                r = Z[i, j] - Y[i, j]
                obj += prob.reg_lambda * r * r
        model.setObjective(obj, GRB.MINIMIZE)

        # Logit-margin constraints for the output layer.  When
        # ``margin_slack_penalty`` is positive, slack variables are added
        # so the problem stays feasible even when the critical-row mask
        # is too restrictive to satisfy every margin.
        if prob.labels is not None and n_out > 1:
            use_slack = prob.margin_slack_penalty > 0
            slack_vars: list[gp.Var] = []
            for i in range(N):
                yi = int(prob.labels[i])
                for j in range(n_out):
                    if j == yi:
                        continue
                    if use_slack:
                        s = model.addVar(lb=0.0, name=f"slack_{i}_{j}")
                        slack_vars.append(s)
                        model.addConstr(Z[i, yi] - Z[i, j] + s >= prob.margin)
                    else:
                        model.addConstr(Z[i, yi] - Z[i, j] >= prob.margin)
            if use_slack and slack_vars:
                obj += prob.margin_slack_penalty * gp.quicksum(slack_vars)
                model.setObjective(obj, GRB.MINIMIZE)

        model.optimize()

        if model.SolCount == 0:
            raise RuntimeError(
                f"Gurobi did not return a feasible solution (status={model.Status})."
            )

        W_out = np.empty_like(W0)
        for (j, k), var in W_new.items():
            W_out[j, k] = var.X
        return W_out


# ---------------------------------------------------------------------------
# SDP backend
# ---------------------------------------------------------------------------


class SDPRepair:
    """Convex SDP repair via CVXPY.

    Solves a continuous relaxation of the repair problem with a Lipschitz
    upper bound on the repaired layer.  After solving, the continuous
    weights are projected onto the quantisation grid so the result remains
    deployable.
    """

    def __init__(
        self,
        solver: Optional[str] = None,
        verbose: bool = False,
        project_to_grid: bool = True,
    ):
        self.solver = solver
        self.verbose = verbose
        self.project_to_grid = project_to_grid

    # ------------------------------------------------------------------
    def solve(self, prob: RepairProblem) -> np.ndarray:
        try:
            import cvxpy as cp
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "SDPRepair requires the `cvxpy` package. Install it with "
                "`pip install cvxpy`."
            ) from exc

        W0 = prob.W0
        b0 = prob.b0
        A = prob.A
        Y = prob.Y_target
        n_out, n_in = W0.shape
        crit_mask = _row_mask(n_out, prob.critical_indices)

        # Free variables only for critical rows; frozen rows are constants.
        Delta = cp.Variable((n_out, n_in))

        constraints: list[cp.Constraint] = []
        # Freeze non-critical rows.
        for j in range(n_out):
            if not crit_mask[j]:
                constraints.append(Delta[j, :] == 0)

        if prob.delta_inf_bound is not None:
            constraints.append(cp.norm(Delta, "inf") <= prob.delta_inf_bound)

        W_new = W0 + Delta

        # Reconstruction loss + Frobenius regulariser.
        recon = cp.sum_squares(A @ W_new.T + np.broadcast_to(b0, (A.shape[0], n_out)) - Y)
        reg = cp.sum_squares(Delta)
        objective = cp.Minimize(reg + prob.reg_lambda * recon)

        # Lipschitz / spectral norm bound via Schur-complement LMI.
        if prob.lipschitz_bound is not None:
            L = float(prob.lipschitz_bound)
            top = cp.hstack([L * np.eye(n_out), W_new])
            bot = cp.hstack([W_new.T, L * np.eye(n_in)])
            M = cp.vstack([top, bot])
            constraints.append(M >> 0)

        # Logit-margin constraints for the output layer (linear in W).
        # When ``margin_slack_penalty > 0`` use a soft hinge formulation
        # so the SDP stays feasible if the critical mask is too tight.
        if prob.labels is not None and n_out > 1:
            N = A.shape[0]
            slack = cp.Variable((N, n_out), nonneg=True)
            for i in range(N):
                yi = int(prob.labels[i])
                z = A[i] @ W_new.T + b0
                for j in range(n_out):
                    if j == yi:
                        continue
                    if prob.margin_slack_penalty > 0:
                        constraints.append(
                            z[yi] - z[j] + slack[i, j] >= prob.margin
                        )
                    else:
                        constraints.append(z[yi] - z[j] >= prob.margin)
            if prob.margin_slack_penalty > 0:
                objective = cp.Minimize(
                    reg
                    + prob.reg_lambda * recon
                    + prob.margin_slack_penalty * cp.sum(slack)
                )

        problem = cp.Problem(objective, constraints)
        solver = self.solver
        if solver is None:
            solver = "SCS" if prob.lipschitz_bound is not None else "OSQP"
        problem.solve(solver=solver, verbose=self.verbose)

        if Delta.value is None:
            raise RuntimeError(
                f"CVXPY failed to solve the SDP (status={problem.status})."
            )

        W_out = W0 + Delta.value
        if self.project_to_grid:
            W_out = self._project_to_grid(W_out, prob.weight_cfg)
            # Keep frozen rows exactly equal to the original.
            for j in range(n_out):
                if not crit_mask[j]:
                    W_out[j, :] = W0[j, :]
        return W_out

    # ------------------------------------------------------------------
    @staticmethod
    def _project_to_grid(W: np.ndarray, cfg: QuantConfig) -> np.ndarray:
        qmin, qmax = cfg.qmin_qmax()
        scale = float(max(np.abs(W).max(), 1e-8) / qmax)
        q = np.clip(np.round(W / scale), qmin, qmax)
        return q * scale
