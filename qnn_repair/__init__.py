"""qnn_repair
=================

End-to-end framework that mirrors the architecture shown in the project
figure:

    [Quantized NN + clean / adversarial inputs]
        --> [Neuron importance scoring (1st-order, Hessian, perturbation)]
        --> [Critical neuron set S_crit]
        --> [Repair via SDP or Gurobi MILP]
        --> [Optimised / repaired model]   (closed-loop)

The package is organised in three layers that map 1:1 to the three coloured
panels in the figure:

* :mod:`qnn_repair.quant`   -- quantised models, fake-quantisation noise,
  clean and adversarial input generation.
* :mod:`qnn_repair.importance`  -- neuron importance estimators
  (Taylor / integrated gradient, Hessian-trace / spectrum, activation and
  output-sensitivity perturbation) and a fusion module.
* :mod:`qnn_repair.repair`  -- the actual model-repair backends: a Gurobi
  mixed-integer program and a CVXPY semidefinite program.  Both share a
  common :class:`RepairProblem` description so they are interchangeable.

A high level driver :class:`qnn_repair.pipeline.RepairPipeline` wires the
three stages together (the "iterative closed loop" arrow at the bottom of
the figure).
"""

from .quant import (
    FakeQuantize,
    QuantConfig,
    QuantizedMLP,
    quantize_tensor,
)
from .attacks import fgsm, pgd, cw
from .importance import (
    GradientImportance,
    HessianImportance,
    PerturbationImportance,
    ImportanceFusion,
    select_critical_neurons,
)
from .repair import (
    RepairProblem,
    GurobiRepair,
    SDPRepair,
)
from .pipeline import RepairPipeline

__all__ = [
    "FakeQuantize",
    "QuantConfig",
    "QuantizedMLP",
    "quantize_tensor",
    "fgsm",
    "pgd",
    "cw",
    "GradientImportance",
    "HessianImportance",
    "PerturbationImportance",
    "ImportanceFusion",
    "select_critical_neurons",
    "RepairProblem",
    "GurobiRepair",
    "SDPRepair",
    "RepairPipeline",
]
