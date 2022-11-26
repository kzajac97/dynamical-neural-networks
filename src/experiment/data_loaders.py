from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch.utils.data

from src.utils.datasets.sequences import generate_time_series_windows, time_series_train_test_split
from src.utils.iterables import filter_dict


class AbstractDataLoader(ABC):
    """Interface for DataLoader"""

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]):
        ...

    @abstractmethod
    def get_training_data(self) -> Iterable:
        ...

    @abstractmethod
    def get_test_data(self) -> Iterable:
        ...


class TorchTimeSeriesCsvDataLoader(AbstractDataLoader):
    def __init__(
        self,
        dataset_path: Path,
        input_columns: list[str],
        output_columns: list[str],
        window_generation_config: dict[str, int],
        test_size: Union[int, float] = 0.5,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param dataset_path: path to CSV file with dataset
        :param input_columns: names of columns containing systems inputs
        :param output_columns: names of columns containing systems outputs
        :param window_generation_config: dict with window generation configuration, it can contain empty values
                                         which will results in using defaults from `generate_time_series_windows`
        :param test_size: test size in samples or ration, for details see `time_series_train_test_split`
        :param batch_size: batch size used to train and test the model by torch DataLoaders
        :param dtype: tensor data type, must much model data type in experiment, defaults to torch.float32
        """
        self.dataset_path = dataset_path
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.test_size = test_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.window_generation_config = window_generation_config

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        # TODO: Implement as required by SweepRunner
        ...

    @cached_property
    def dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path)

    @cached_property
    def train_and_test_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Pre-loaded dataset split into train and test"""
        return time_series_train_test_split(self.dataset, test_size=self.test_size)

    @cached_property
    def train_dataset(self) -> pd.DataFrame:
        return self.train_and_test_datasets[0]

    @cached_property
    def test_dataset(self) -> pd.DataFrame:
        return self.train_and_test_datasets[1]

    def generate_windows(self, inputs: Optional[Sequence], outputs: Sequence) -> dict[str, Sequence]:
        """
        Use generate_time_series_windows with given configuration to generate windows used by the model

        Always generates using order convention, which must be handled by the model:
            * backward_inputs
            * backward_outputs
            * forward_inputs
            * forward_outputs
        """
        parameters = dict(
            outputs=outputs,
            inputs=inputs,
            shift=self.window_generation_config.get("shift"),
            forward_input_window_size=self.window_generation_config.get("forward_input_window_size"),
            forward_output_window_size=self.window_generation_config.get("forward_output_window_size"),
            backward_input_window_size=self.window_generation_config.get("backward_input_window_size"),
            backward_output_window_size=self.window_generation_config.get("backward_output_window_size"),
            forward_input_mask=self.window_generation_config.get("forward_input_mask"),
            forward_output_mask=self.window_generation_config.get("forward_output_mask"),
            backward_input_mask=self.window_generation_config.get("backward_input_mask"),
            backward_output_mask=self.window_generation_config.get("backward_output_mask"),
        )

        return generate_time_series_windows(**filter_dict(None, parameters))

    def get_training_data(self) -> Iterable:
        """Generates training data and returns torch DataLoader"""
        inputs = self.train_dataset[self.input_columns].values if self.input_columns else None
        outputs = self.train_dataset[self.output_columns].values

        train_windows = self.generate_windows(inputs, outputs)
        tensors = [torch.from_numpy(window).to(self.dtype) for window in train_windows.values() if len(window) != 0]

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors), batch_size=self.batch_size, shuffle=True
        )

    def get_test_data(self) -> Iterable:
        """Generates training data and returns torch DataLoader"""
        inputs = self.test_dataset[self.input_columns].values if self.input_columns else None
        outputs = self.test_dataset[self.output_columns].values

        test_windows = self.generate_windows(inputs, outputs)
        tensors = [torch.from_numpy(window).to(self.dtype) for window in test_windows.values() if len(window) != 0]

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*tensors), batch_size=self.batch_size, shuffle=False
        )
