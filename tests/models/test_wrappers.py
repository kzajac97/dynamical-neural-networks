import numpy as np
import pytest
import torch

from src.models.wrappers import ResidualWrapper


@pytest.fixture
def mock_torch_module():
    class MockModule(torch.nn.Module):
        """Mock module, which performs adds 1 to all inputs"""

        def __init__(self):
            super(MockModule, self).__init__()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return 2 * inputs

    yield MockModule()


@pytest.mark.parametrize(
    "inputs, expected_outputs",
    [
        (torch.zeros(16, 10, 1), torch.zeros(16, 10, 1)),
        (torch.ones(12, 10, 5), torch.full(size=(12, 10, 5), fill_value=3.0)),
        (torch.arange(10), torch.arange(0, 30, 3)),
    ],
)
def test_wrapper(inputs: torch.Tensor, expected_outputs: torch.Tensor, mock_torch_module):
    model = ResidualWrapper(mock_torch_module)
    outputs = model(inputs)

    np.testing.assert_array_equal(outputs.numpy(), expected_outputs.numpy())
