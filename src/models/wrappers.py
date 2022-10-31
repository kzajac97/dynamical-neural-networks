import torch


class ResidualWrapper(torch.nn.Module):
    """
    Wrapper converting any torch Module into residual module
    Useful for time-series, when whole network is wrapped unlike skip-connection like in ResNet
    """

    def __init__(self, module: torch.nn.Module):
        super(ResidualWrapper, self).__init__()
        self.module = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        deltas = self.module(inputs)
        return inputs + deltas
