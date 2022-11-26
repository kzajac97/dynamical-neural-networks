from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch

from src import utils
from src.metrics.regression import regression_score
from src.models.utils import count_trainable_parameters


class AbstractReporter(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Can't call abstract method!")


class RepressionReporter(AbstractReporter):
    def __init__(self):
        ...

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
        self, model: torch.nn.Module, targets: torch.Tensor, predictions: torch.Tensor, training_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """
        :param model: trained model
        :param targets: targets for the test set
        :param predictions: model predictions on the test set
        :param training_summary: training summary from trainer

        :return: dict of reported metrics to log into WANDB
        """
        metrics = self.compute_regression_metrics(targets, predictions)
        model_summary = self.get_model_summary(model)

        return dict(**metrics, **training_summary, **model_summary)
