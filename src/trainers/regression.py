import dataclasses
from timeit import default_timer as timer
from typing import Iterable, Tuple

import torch.nn

from src.trainers.callbacks import BaseCallback, CallbackParameters
from src.utils.exceptions import StopTraining
from src.utils.types import PrintFunction


class TimeSeriesRegressionTrainer:
    """
    Simple class for training time-series regression models using pytorch

    It requires passing initialized model, optimizer and loss function wit additional parameters
    It supports stopping training by raising StopTraining exception and callbacks (see `src.experiments.callbacks.py`)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: torch.nn.modules.loss._Loss,
        callbacks: list[BaseCallback],
        n_epochs: int,
        device: str = "cpu",
        print_fn: PrintFunction = print,
    ):
        """
        :param model: model to train
        :param optimizer: optimizer to use for training
        :param loss_function: loss function
        :param callbacks:
        :param n_epochs:
        :param device: torch device, can be `cuda` or `cpu`
        :param print_fn: function to log information about training state, defaults to standard print
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.callbacks = callbacks
        self.n_epochs = n_epochs
        self.device = device
        self.print_fn = print_fn
        self.training_summary = {}

    def predict(self, data_loader: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        targets, predictions = [], []

        for inputs, y_true in data_loader:
            with torch.no_grad():
                inputs = inputs.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.model(inputs)

            predictions.append(y_pred)
            targets.append(y_true)

        return torch.cat(predictions), torch.cat(targets)

    def train_epoch(self, data_loader: Iterable) -> None:
        self.model.train()
        for x, y in data_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x)

            loss_value = self.loss_function(y_pred, y)
            loss_value.backward()
            self.optimizer.step()

    def callback_loop(self, parameters: CallbackParameters) -> None:
        for callback in self.callbacks:
            callback(**dataclasses.asdict(parameters))

    def train(self, data_loader: Iterable) -> torch.nn.Module:
        try:
            start_time = timer()
            for epoch in range(self.n_epochs):
                self.train_epoch(data_loader)
                targets, predictions = self.predict(data_loader)  # evaluate using training data to log metrics

                self.callback_loop(
                    parameters=CallbackParameters(
                        epoch=epoch,
                        model=self.model,
                        optimizer=self.optimizer,
                        loss_function=self.loss_function,
                        targets=targets,
                        predictions=predictions,
                    )
                )

        except StopTraining:
            end_time = timer()
            self.training_summary = {"epochs": epoch, "training_time": end_time - start_time}
            self.print_fn(f"Stopping training at {epoch}...")
            return self.model

        end_time = timer()
        self.training_summary = {"epochs": epoch, "training_time": end_time - start_time}
        return self.model
