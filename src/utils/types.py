from typing import Any, Callable, TypeVar, Literal

import torch

PrintFunction = TypeVar("PrintFunction", bound=Callable[[str], None])
TorchParameterizedModel = TypeVar("TorchParameterizedModel", bound=Callable[[dict[str, Any]], torch.nn.Module])
TorchDevice = Literal["cpu", "device"]
