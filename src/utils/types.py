from typing import Any, Callable, TypeVar

import torch

PrintFunction = TypeVar("PrintFunction", bound=Callable[[str], None])
TorchReportFunction = TypeVar(
    "TorchReportFunction", bound=Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], dict[str, float]]
)
TorchParameterizedModel = TypeVar("TorchParameterizedModel", bound=Callable[[dict[str, Any]], torch.nn.Module])
