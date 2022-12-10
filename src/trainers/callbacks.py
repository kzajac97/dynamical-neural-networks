from abc import ABC, abstractmethod
from functools import cached_property
from timeit import default_timer as timer
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src import utils
from src.utils.exceptions import StopTraining


class AbstractCallback(ABC):
    def __init__(self, run_period: int):
        """
        :param run_period: run frequency in epochs
        """
        self.run_period = run_period

    @abstractmethod
    def __call__(self, epoch: int, metrics: pd.Series) -> None:
        raise NotImplementedError("Cannot call BaseCallback directly!")


class EarlyStoppingCallback(AbstractCallback):
    """
    EarlyStoppingCallback monitors chosen metric (using src.metrics.regression.regression_score keys)
    to send signal to trained when model should stop training based on moving average of values of this metric
    If moving average with certain window size is not decreasing with respect to some min_delta training is stopped

    The stop signal is send using StopTraining exception, which is subclass of StopIteration
    and it must be handled by training function or Trainer class.
    """

    def __init__(
        self,
        metric_name: str,
        moving_average_window_size: int,
        run_period: int = 1,
        patience: Optional[int] = None,
        delta: float = 0.0,
    ):
        """
        :param metric_name: the metric to use for early stopping
        :param moving_average_window_size: window size used in moving average computation
        :param run_period: run frequency in epochs, defaults to 1
        :param patience: number of epochs to wait before stopping
        :param delta: maximal amount on increase in moving average computation, defaults to zero
                      for positive values this allows values of metric to increase
                      for negative values this forces values of metric to decrease by given value
        """
        super().__init__(run_period)

        self.patience = patience
        self.window_size = moving_average_window_size
        self.metric_name = metric_name
        self.delta = delta

        self.metric_values = []

    @cached_property
    def patience_slice(self) -> slice:
        return slice(-1 * self.patience, None) if self.patience else slice(None)

    def should_stop(self) -> bool:
        """Returns True if the model should stop training."""
        if len(self.metric_values) <= self.window_size:
            return False  # do not stop in first `window_size` epochs

        metric_values = np.asarray(self.metric_values[self.patience_slice])  # noqa
        metric_average = utils.numpy.moving_average(metric_values, window_size=self.window_size)
        return np.any(np.diff(metric_average) > self.delta)

    def __call__(self, epoch: int, metrics: pd.Series) -> None:
        self.metric_values.append(metrics[self.metric_name])

        if self.should_stop():
            raise StopTraining  # early stopping reached!


class TrainingTimeoutCallback(AbstractCallback):
    """
    Callback checks if training exceeded maximal time in seconds
    after each epoch (or multiple epochs, if larger `run_frequency given)
    """

    def __init__(self, max_training_time: int, run_period: int = 1):
        """
        :param max_training_time: maximal allowed training time in seconds
        :param run_period: run frequency in epochs, defaults to 1 for larger values
        """
        super().__init__(run_period)

        self.max_training_time = max_training_time
        self.start_time = None

    def __call__(self, epoch: int, metrics: pd.Series) -> None:
        if self.start_time is None:
            self.start_time = timer()

        current_time = timer()
        if current_time - self.start_time >= self.max_training_time:
            raise StopTraining


class RegressionReportCallback(AbstractCallback):
    """Callback prints regression metrics at each epoch"""

    def __init__(
        self,
        run_period: int = 1,
        print_fn: Callable[[str], None] = print,
        metric_names: Optional[list[str]] = slice(None),
        precision: int = 4,
        width: int = 4,
        separator: str = " ",
    ):
        """
        :param run_period: run frequency in epochs, defaults to 1
        :param print_fn: function to print metrics, defaults to print but loggers or file streams can be used as well
        :param metric_names: Metrics to log, if None logs all regression metrics (see `src.metrics.regression`)
        :param precision: Precision to use when logging the metrics
        :param width: Spacing between logged metrics
        :param separator: The separator to use between logged metrics
        """
        super().__init__(run_period)

        self.print_fn = print_fn
        self.metric_names = metric_names
        self.precision = precision
        self.width = width
        self.separator = separator

    def _format_metrics(self, metrics: pd.Series) -> str:
        """Formats the metrics for logging."""
        spacing = self.separator * self.width

        return f"{spacing}".join(
            [f"{metric_name}: {metric:.{self.precision}f}" for metric_name, metric in metrics.to_dict().items()]
        )

    def __call__(self, epoch: int, metrics: pd.Series) -> None:
        """Prints all regression metrics as each epoch"""
        self.print_fn(f"Epoch: {epoch}{self.separator * self.width}{self._format_metrics(metrics[self.metric_names])}")


class CallbackList:
    """Abstract aggregator for callbacks"""

    def __init__(self, callbacks: list[AbstractCallback]):
        """
        :param callbacks: list of created callbacks
        """
        self.callbacks = callbacks

    def __call__(self, epoch: int, metrics: pd.Series) -> None:
        for callback in self.callbacks:
            if epoch % callback.run_period == 0:
                callback(epoch, metrics)
