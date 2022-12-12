import uuid
from timeit import default_timer as timer
from typing import Iterable, Optional, Tuple

import pandas as pd
import torch.nn

from src import utils
from src.metrics.regression import regression_score
from src.trainers.callbacks import CallbackHandler
from src.trainers.checkpoints import CheckpointHandler
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
        callback_handler: CallbackHandler,
        checkpoint_handler: CheckpointHandler,
        n_epochs: int,
        name: Optional[str] = None,
        device: str = "cpu",
        print_fn: PrintFunction = print,
    ):
        """
        :param model: model to train
        :param optimizer: optimizer to use for training
        :param loss_function: loss function
        :param callback_handler: list of created callbacks
        :param n_epochs: number of epochs to train for (can be stopped early by callbacks)
        :param name: name of the training run, used to distinguish runs when using SweepRunner
        :param device: torch device, can be `cuda` or `cpu`
        :param print_fn: function to log information about training state, defaults to standard print
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.callback_handler = callback_handler
        self.checkpoint_handler = checkpoint_handler
        self.n_epochs = n_epochs
        self.device = device
        self.print_fn = print_fn

        self.name = name if name is not None else uuid.uuid4().hex
        self.training_summary = {}

    @staticmethod
    def evaluate(targets: torch.Tensor, predictions: torch.Tensor) -> pd.Series:
        """Evaluates regression metrics for given targets and predictions"""
        targets = utils.tensors.torch_to_flat_array(targets)
        predictions = utils.tensors.torch_to_flat_array(predictions)
        return regression_score(y_true=targets, y_pred=predictions)

    def predict(self, data_loader: Iterable) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict on given data and return targets and predictions as torch"""
        self.model.eval()
        targets, predictions = [], []

        for inputs, y_true in data_loader:
            with torch.no_grad():
                inputs = inputs.to(self.device)
                y_true = y_true.to(self.device)

                y_pred = self.model(inputs)

            predictions.append(y_pred)
            targets.append(y_true)

        return torch.cat(targets), torch.cat(predictions)

    def train_epoch(self, data_loader: Iterable) -> None:
        """Run single epoch of training"""
        self.model.train()
        for x, y in data_loader:
            x = x.to(self.device)  # casting in the loop to save GPU memory
            y = y.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x)

            loss_value = self.loss_function(y_pred, y)
            loss_value.backward()
            self.optimizer.step()

    def post_train(self) -> torch.nn.Module:
        """Run post-training model processing"""
        if self.checkpoint_handler:
            return self.checkpoint_handler.restore(self.model)

        return self.model

    def train(self, data_loader: Iterable, validation_data_loader: Optional[Iterable] = None) -> None:
        """
        Runs model training with simple loop:
            1. Train single epoch on train dataset
            2. Predict on validation data, if given or on train data
            3. Compute metrics on predictions obtained step before
            4. Run callbacks loop
            5. Run checkpoints save step

        At any point when exception StopTraining is raised training stops (designed to be used by callbacks)
        """
        start_time = timer()
        epoch = 0
        self.print_fn(f"Starting training run {self.name}...")
        for epoch in range(self.n_epochs):
            try:
                self.train_epoch(data_loader)
                # evaluate using training data when validation data not given
                loader = validation_data_loader if validation_data_loader else data_loader
                targets, predictions = self.predict(loader)
                # compute metrics and run callback reporting and checkpointing loops
                metrics = self.evaluate(targets, predictions)
                self.callback_handler(epoch, metrics)
                self.checkpoint_handler.save(self.model, epoch, metrics)

            except StopTraining:
                self.training_summary = {"epochs": epoch, "training_time": timer() - start_time}
                self.print_fn(f"Stopping training at {epoch}...")
                return

        self.print_fn("Stopping training after reaching max epochs...")
        self.training_summary = {"epochs": epoch, "training_time": timer() - start_time}
