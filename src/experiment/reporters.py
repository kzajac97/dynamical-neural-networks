from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
import plotly.express as px
import torch
import wandb

from src import utils
from src.metrics.regression import regression_score
from src.models.utils import count_trainable_parameters


class AbstractReporter(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Can't call abstract method!")


class RepressionReporter(AbstractReporter):
    @staticmethod
    def compute_regression_metrics(targets: torch.Tensor, predictions: torch.Tensor) -> Mapping:
        """Compute regression score using `src.metrics`"""
        targets = utils.tensors.torch_to_flat_array(targets)
        predictions = utils.tensors.torch_to_flat_array(predictions)

        return regression_score(y_true=targets, y_pred=predictions).to_dict()

    @staticmethod
    def get_model_summary(model: torch.nn.Module) -> dict[str, Any]:
        """Get summary of the model parameters"""
        n_trainable_parameters = count_trainable_parameters(model)
        return {"n_trainable_parameters": n_trainable_parameters}

    def __call__(
        self,
        model: torch.nn.Module,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        training_summary: dict[str, Any],
        **kwargs
    ) -> None:
        """
        :param model: trained model
        :param targets: targets for the test set
        :param predictions: model predictions on the test set
        :param training_summary: training summary from trainer

        :return: dict of reported metrics to log into WANDB
        """
        metrics = self.compute_regression_metrics(targets, predictions)
        model_summary = self.get_model_summary(model)

        wandb.log(dict(**metrics, **training_summary, **model_summary))


class PredictionPlotReporter(AbstractReporter):
    """Reporter logging predictions of regression model with targets and error"""

    def __call__(self, targets: torch.Tensor, predictions: torch.Tensor, **kwargs: dict) -> None:
        """
        Logs simple regression plot to WANDB Sweep interface
        containing plot of real values, model predictions and error
        """
        targets = targets.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()

        for state_dim in range(targets.shape[-1]):
            dim_targets = targets[:, :, state_dim].flatten()
            dim_predictions = predictions[:, :, state_dim].flatten()
            error = np.abs(dim_targets - dim_predictions)

            figure = px.line(
                utils.numpy.stack_arrays(
                    arrays=[dim_targets, dim_predictions, error], names=["targets", "predictions", "error"]
                )
            )

            wandb.log({"prediction_plot": figure})


class ReporterList:
    """Abstract wrapper for running multiple reporters"""

    def __init__(self, reporters: list[AbstractReporter]):
        """
        :param reporters: list of reporter AbstractReporter subclass instances
        """
        self.reporters = reporters

    def __call__(self, **parameters: dict[str, Any]) -> None:
        for reporter in self.reporters:
            reporter(**parameters)
