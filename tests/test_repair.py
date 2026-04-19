import numpy as np
import torch

from qnn_repair.attacks import fgsm
from qnn_repair.importance import (
    GradientImportance,
    ImportanceFusion,
    select_critical_neurons,
)
from qnn_repair.quant import QuantConfig, QuantizedMLP
from qnn_repair.repair import (
    GurobiRepair,
    RepairProblem,
    SDPRepair,
    apply_repair_to_model,
)


def _build_repair_problem(seed: int = 0) -> tuple[QuantizedMLP, RepairProblem, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = QuantizedMLP(in_features=4, hidden=[6], out_features=2, bits=[4, 4])
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    x_adv = fgsm(model, x, y, eps=0.05, clip=(-3.0, 3.0))

    grad = GradientImportance().score(model, 0, [(x, y)])
    fused = ImportanceFusion(1.0, 0.0, 0.0)(grad, torch.zeros_like(grad), torch.zeros_like(grad))
    crit = select_critical_neurons(fused, k=3)

    prob = RepairProblem.from_model(
        model,
        layer_idx=0,
        x_clean=x,
        x_adv=x_adv,
        labels=None,
        critical_indices=crit,
        reg_lambda=1.0,
    )
    return model, prob, x, y


def test_sdp_repair_runs_and_freezes_noncritical_rows():
    model, prob, _, _ = _build_repair_problem()
    solver = SDPRepair(project_to_grid=False)
    W_new = solver.solve(prob)
    assert W_new.shape == prob.W0.shape
    crit_set = set(prob.critical_indices.tolist())
    for j in range(prob.W0.shape[0]):
        if j not in crit_set:
            assert np.allclose(W_new[j], prob.W0[j], atol=1e-6)


def test_sdp_lipschitz_bound_is_respected():
    model, prob, _, _ = _build_repair_problem(seed=1)
    prob.lipschitz_bound = 1.0
    prob.reg_lambda = 0.01
    W_new = SDPRepair(project_to_grid=False).solve(prob)
    sigma = np.linalg.svd(W_new, compute_uv=False)[0]
    assert sigma <= 1.0 + 1e-3


def test_apply_repair_writes_back():
    model, prob, _, _ = _build_repair_problem()
    W_new = SDPRepair(project_to_grid=False).solve(prob)
    apply_repair_to_model(model, 0, W_new)
    actual = model.hidden_layer(0).linear.weight.detach().cpu().numpy()
    assert np.allclose(actual, W_new, atol=1e-5)


def test_gurobi_repair_optional():
    try:
        import gurobipy  # noqa: F401
    except Exception:
        return  # Gurobi not installed/licensed; skip.
    model, prob, _, _ = _build_repair_problem(seed=2)
    # Use the continuous QP to stay inside the size-limited free license.
    W_new = GurobiRepair(integer=False, time_limit=10.0).solve(prob)
    assert W_new.shape == prob.W0.shape
