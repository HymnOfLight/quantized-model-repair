import torch

from qnn_repair.quant import QuantConfig, QuantizedMLP, quantize_tensor


def test_quantize_tensor_round_trip_within_grid():
    cfg = QuantConfig(bits=4, signed=True, symmetric=True)
    x = torch.linspace(-1.0, 1.0, 16)
    y = quantize_tensor(x, cfg)
    qmin, qmax = cfg.qmin_qmax()
    scale = max(x.abs().max().item(), 1e-8) / qmax
    levels = scale * torch.arange(qmin, qmax + 1).to(x)
    diffs = (y.unsqueeze(-1) - levels).abs().min(dim=-1).values
    assert torch.all(diffs < 1e-5)


def test_quantized_mlp_forward_shapes():
    model = QuantizedMLP(in_features=4, hidden=[8, 6], out_features=3, bits=[8, 4, 2])
    x = torch.randn(5, 4)
    out = model(x)
    assert out.shape == (5, 3)
    out2, acts = model.forward_with_activations(x)
    assert torch.allclose(out, out2)
    assert len(acts) == 2
    assert acts[0].shape == (5, 8)
    assert acts[1].shape == (5, 6)


def test_quantized_mlp_grad_flow():
    model = QuantizedMLP(in_features=3, hidden=[5], out_features=2, bits=[4, 4])
    x = torch.randn(4, 3, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)
