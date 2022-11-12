import pytest
import torch.optim

from src.experiment.config import build_loss_function, build_optimizer


@pytest.mark.parametrize(
    "name, parameters",
    [
        ("adadelta", {}),
        ("adagrad", {"lr": 0.01, "lr_decay": 0.001}),
        ("adam", {}),
        ("adamw", {"amsgrad": True}),
        ("sparse_adam", {}),
        ("adamax", {}),
        ("asgd", {"lambd": 0.001, "t0": 1000}),
        ("lbfgs", {}),
        ("nadam", {"betas": (0.9, 0.99)}),
        ("radam", {}),
        ("rmsprop", {"alpha": 0.9, "centered": True}),
        ("rprop", {}),
        ("sgd", {"lr": 0.1}),
    ],
)
def test_build_optimizer(name: str, parameters: dict, mock_torch_module):
    optimizer = build_optimizer(mock_torch_module, name, parameters)
    assert isinstance(optimizer, torch.optim.Optimizer)


@pytest.mark.parametrize(
    "name, parameters",
    [
        ("sgd", {"param": 0.1}),
        ("sgd", {"param_x": 0.1, "param_y": 5}),
        ("adam", {"param_x": 0.1}),
    ],
)
def test_build_optimizer_with_invalid_parameters(name: str, parameters: dict, mock_torch_module):
    with pytest.raises(TypeError):
        _ = build_optimizer(mock_torch_module, name, parameters)


@pytest.mark.parametrize(
    "name, parameters",
    [
        ("adamx", {}),
        ("optimizer", {}),
        ("optm", {}),
    ],
)
def test_build_optimizer_invalid_key(name: str, parameters: dict, mock_torch_module):
    with pytest.raises(AssertionError):
        _ = build_optimizer(mock_torch_module, name, parameters)


@pytest.mark.parametrize(
    "name, parameters",
    [
        ("mse", {}),
        ("l1", {"reduction": "sum"}),
        ("huber", {"delta": 0.8}),
        ("smooth_l1", {}),
    ],
)
def test_build_loss_function(name: str, parameters: dict):
    loss_fn = build_loss_function(name, parameters)
    assert isinstance(loss_fn, torch.nn.modules.loss._Loss)
