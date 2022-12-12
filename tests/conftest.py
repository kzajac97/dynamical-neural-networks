import pytest
import torch


@pytest.fixture
def mock_torch_module():
    class MockModule(torch.nn.Module):
        """Mock module, which performs adds 1 to all inputs"""

        def __init__(self):
            super(MockModule, self).__init__()
            self.param = torch.nn.Parameter(torch.zeros(10))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return 2 * inputs

    yield MockModule()
