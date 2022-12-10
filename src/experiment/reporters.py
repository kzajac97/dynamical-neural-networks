from abc import ABC, abstractmethod
from typing import Any, Mapping

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
        targets = utils.tensors.torch_to_flat_array(targets)
        predictions = utils.tensors.torch_to_flat_array(predictions)

        return regression_score(y_true=targets, y_pred=predictions).to_dict()

    @staticmethod
    def get_model_summary(model: torch.nn.Module) -> dict[str, Any]:
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
        targets = utils.tensors.torch_to_flat_array(targets)
        predictions = utils.tensors.torch_to_flat_array(predictions)

        figure = px.line(
            utils.numpy.stack_arrays(
                arrays=[targets, predictions, targets - predictions], names=["target", "predictions", "error"]
            )
        )

        wandb.log({"prediction_plot": figure})


class ReporterList:
    def __init__(self, reporters: list[AbstractReporter]):
        self.reporters = reporters

    def __call__(self, **parameters: dict[str, Any]) -> None:
        for reporter in self.reporters:
            reporter(**parameters)
