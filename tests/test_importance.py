import torch

from qnn_repair.importance import (
    GradientImportance,
    HessianImportance,
    ImportanceFusion,
    PerturbationImportance,
    select_critical_neurons,
)
from qnn_repair.quant import QuantizedMLP


def _toy_problem():
    torch.manual_seed(0)
    model = QuantizedMLP(in_features=4, hidden=[6, 5], out_features=3, bits=[4, 4, 4])
    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))
    return model, [(x, y)]


def test_gradient_importance_shape_and_finiteness():
    model, loader = _toy_problem()
    s = GradientImportance("taylor").score(model, layer_idx=0, loader=loader)
    assert s.shape == (6,)
    assert torch.isfinite(s).all()


def test_hessian_importance_shape():
    model, loader = _toy_problem()
    s = HessianImportance(num_samples=2).score(model, 0, loader)
    assert s.shape == (6,)
    assert torch.isfinite(s).all()


def test_perturbation_importance_modes():
    model, loader = _toy_problem()
    s_act = PerturbationImportance("activation").score(model, 0, loader)
    s_occ = PerturbationImportance("occlusion").score(model, 0, loader)
    assert s_act.shape == (6,)
    assert s_occ.shape == (6,)


def test_fusion_and_topk():
    fused = ImportanceFusion(1.0, 1.0, 1.0)(
        torch.tensor([0.1, 0.9, 0.5]),
        torch.tensor([0.0, 0.5, 1.0]),
        torch.tensor([1.0, 0.0, 0.5]),
    )
    idx = select_critical_neurons(fused, k=2)
    assert idx.tolist() == [1, 2]
