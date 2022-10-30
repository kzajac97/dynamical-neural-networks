from typing import Callable, Set

import numpy as np
import pytest
import torch

from src.models.transforms import ComplexForwardFourier, RealBackwardFourier, RealForwardFourier
from src.models.utils import count_trainable_parameters


@pytest.fixture
def time_array():
    return np.linspace(0, 4 * np.pi, 100, endpoint=False)


@pytest.mark.parametrize(
    "function_set",
    [
        ({np.sin}),
        ({np.sin, np.cos}),
        {np.sin, np.cos, lambda t: np.sin(2 * t)},
        {lambda t: t * 2},
        {lambda t: np.zeros(t.shape)},
    ],
)
def test_real_fourier_transforms(function_set: Set[Callable], time_array):
    """
    Test checks if applying forward and backward Fourier transforms (implemented as torch.modules)
    returns the same input with precision up to 3 decimal places
    """
    inputs = np.column_stack([f(time_array) for f in function_set])
    inputs = torch.from_numpy(inputs).unsqueeze(dim=0)  # convert to batch with single tensor

    model = torch.nn.Sequential(
        RealForwardFourier(),
        RealBackwardFourier(),
    )

    outputs = model(inputs)
    np.testing.assert_almost_equal(inputs.numpy().squeeze(), outputs.numpy().squeeze(), decimal=3)


@pytest.mark.parametrize("model", [RealForwardFourier, ComplexForwardFourier, RealBackwardFourier])
def test_non_trainable(model: torch.nn.Module):
    """Tests if transforms implemented as modules are not-trainable"""
    assert count_trainable_parameters(model()) == 0
