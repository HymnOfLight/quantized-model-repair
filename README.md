# quantized-model-repair

End-to-end framework that mirrors the architecture in the project figure:

> **Quantised neural network + clean / adversarial inputs**  →  **Neuron
> importance evaluation** (1st-order gradient, 2nd-order Hessian,
> perturbation/output sensitivity)  →  **Critical neuron set
> `S_crit`**  →  **Application: pruning, mixed precision, robustness
> enhancement, interpretability**.

This repository implements the full closed loop, with a particular focus
on the **robustness-enhancement** arrow: given the critical neuron set,
the layer is *repaired* by solving a constrained optimisation problem
through one of two interchangeable backends:

* **`GurobiRepair`** -- mixed-integer quadratic program (MIQP) that
  forces the repaired weights onto the discrete quantisation grid, keeps
  non-critical rows frozen, and adds **hard logit-margin constraints** so
  that adversarial samples are guaranteed to be classified correctly.
* **`SDPRepair`** -- convex semidefinite program (CVXPY) with a
  **Schur-complement LMI** that upper-bounds the spectral norm
  (`σ_max(W) ≤ L`) of the repaired layer, controlling the worst-case
  Lipschitz amplification of input perturbations. Continuous weights are
  optionally projected back to the quantisation grid.

## Mapping to the figure

| Figure panel | Code |
| --- | --- |
| `量化模型` (mixed INT8/INT4/INT2 + activations) | `qnn_repair.quant.QuantizedMLP` with `bits=[8, 4, 2]`, `FakeQuantize` STE |
| `量化噪声 (Noise)` / `离散激活` | per-tensor + per-channel fake quantisers in `quant.py` |
| `输入数据`: 洁净 + 对抗 (FGSM/PGD/CW) | `qnn_repair.attacks.fgsm / pgd / cw` |
| `基于一阶梯度`: Taylor + integrated gradient | `GradientImportance(mode="taylor"/"integrated")` |
| `基于二阶 Hessian`: 迹 + 谱分析 | `HessianImportance` (Hutchinson trace + diag estimator) |
| `基于扰动与输出`: 激活值 + 输出敏感度 | `PerturbationImportance(mode="activation"/"occlusion")` |
| α/β/γ `方法融合与聚合` | `ImportanceFusion(alpha, beta, gamma)` |
| `关键神经元识别` → `S_crit` | `select_critical_neurons(fused_scores, k=...)` |
| `模型剪枝 / 混合精度 / 鲁棒性增强 / 可解释性` | `qnn_repair.repair.GurobiRepair`, `SDPRepair` |
| `指导模型优化与重构（迭代闭环）` | `qnn_repair.pipeline.RepairPipeline.repair_all` |

## Mathematical formulation

For a target layer with weights `W₀ ∈ ℝ^(n_o × n_i)`, bias `b₀`,
calibration activations `A ∈ ℝ^(N × n_i)`, desired pre-activation outputs
`Y ∈ ℝ^(N × n_o)`, and critical-row mask `S ⊂ {1, …, n_o}`:

```
minimise   ‖Δ‖_F²  +  λ · ‖A (W₀ + Δ)ᵀ + 1·b₀ᵀ − Y‖_F²
subject to Δ_{j,:} = 0          for j ∉ S          (freeze non-critical)
           W₀ + Δ on the quant grid                (Gurobi MIQP)
           (W+Δ)_{y_i,:} aᵢ − (W+Δ)_{j,:} aᵢ ≥ m   (Gurobi: margin)
           ‖W₀ + Δ‖₂ ≤ L                           (SDP LMI)
```

The SDP backend implements `‖W‖₂ ≤ L` exactly via the Schur complement

```
[ L·I       (W₀+Δ)ᵀ ]
[ W₀+Δ     L·I       ]  ⪰  0.
```

## Installation

```bash
pip install -r requirements.txt
# Gurobi requires a (free academic / commercial) licence.
# CVXPY ships with the SCS SDP solver, which is sufficient for SDPRepair.
```

## Quickstart

```python
from qnn_repair import (
    ImportanceFusion, QuantizedMLP, RepairPipeline,
)

model = QuantizedMLP(in_features=2, hidden=[16, 8], out_features=2,
                     bits=[8, 4, 2], act_bits=8)
# ... train model on clean data ...

pipeline = RepairPipeline(
    backend="sdp",                # or "gurobi"
    attack="pgd", eps=0.1,
    top_k_ratio=0.4,
    fusion=ImportanceFusion(alpha=1.0, beta=0.5, gamma=0.5),
    lipschitz_bound=2.0,          # SDP only
    margin=0.3,                   # Gurobi only
    verbose=True,
)
reports = pipeline.repair_all(model, calibration_loader)
for r in reports:
    print(r)
```

A complete end-to-end example is provided in
[`examples/demo_repair.py`](examples/demo_repair.py):

```bash
python examples/demo_repair.py --backend sdp --attack pgd --eps 0.1
python examples/demo_repair.py --backend gurobi --top-k 0.5
```

## Tests

```bash
pip install pytest
pytest -q
```

Tests cover the quantiser STE, the three importance estimators, the
SDP repair (objective, frozen-row constraint, Lipschitz LMI), and the
Gurobi backend (skipped automatically if `gurobipy` is unavailable).

## Project layout

```
qnn_repair/
├── __init__.py        # package surface
├── quant.py           # FakeQuantize, QuantizedMLP, QuantConfig
├── attacks.py         # fgsm, pgd, cw
├── importance.py      # GradientImportance, HessianImportance,
│                      # PerturbationImportance, ImportanceFusion
├── repair.py          # RepairProblem, GurobiRepair, SDPRepair
└── pipeline.py        # RepairPipeline (closed-loop driver)
examples/demo_repair.py
tests/
└── test_quant.py, test_importance.py, test_repair.py
```
