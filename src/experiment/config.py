from types import MappingProxyType
from typing import Any

import torch

from src.trainers import callbacks


TORCH_OPTIMIZERS = MappingProxyType(
    {
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sparse_adam": torch.optim.SparseAdam,
        "adamax": torch.optim.Adamax,
        "asgd": torch.optim.ASGD,
        "lbfgs": torch.optim.LBFGS,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD,
    }
)


TORCH_LOSS_FUNCTIONS = MappingProxyType(
    {
        "mse": torch.nn.MSELoss,
        "l1": torch.nn.L1Loss,
        "huber": torch.nn.HuberLoss,
        "smooth_l1": torch.nn.SmoothL1Loss,
    }
)

CALLBACKS = MappingProxyType(
    {
        "early_stopping": callbacks.EarlyStoppingCallback,
        "regression_report": callbacks.RegressionReportCallback,
        "torch_checkpoint": callbacks.TorchCheckpointCallback,
        "training_timeout": callbacks.TrainingTimeoutCallback,
    }
)


def build_optimizer(model: torch.nn.Module, name: str, parameters: dict[str, Any]) -> torch.optim.Optimizer:
    """Builds torch optimizer from available optimizer classes using string accessor"""
    optimizer = TORCH_OPTIMIZERS.get(name)
    assert optimizer, f"Attempting to use non-existing  optimizer! Valid parameters are: {TORCH_OPTIMIZERS.keys()}"
    return optimizer(model.parameters(), **parameters)


def build_loss_function(name, parameters: dict[str, Any]) -> torch.nn.modules.loss._Loss:
    """
    Builds torch loss function from available loss classes using string accessor
    Supports only regression losses!
    """
    loss_fn = TORCH_LOSS_FUNCTIONS.get(name)
    assert loss_fn, f"Attempting to use non-existing loss function! Valid parameters are: {TORCH_LOSS_FUNCTIONS.keys()}"
    return loss_fn(**parameters)


def build_callbacks(names: list[str], parameters: list[dict]) -> list[callbacks.BaseCallback]:
    """
    :param names: list of callback names
    :param parameters: dict of name to callback parameters for given names

    :return: list of build callbacks
    """
    callbacks_list = []

    for name, parameter_set in zip(names, parameters):
        callback = CALLBACKS.get(name)
        assert callback, f"Attempting to use non-existing callback! Supported parameters are: {CALLBACKS.keys()}"
        callbacks_list.append(callback(**parameter_set))

    return callbacks_list
