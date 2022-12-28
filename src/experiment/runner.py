import traceback
from pathlib import Path
from typing import Any

import torch
import wandb

from src.experiment import config
from src.experiment.data_loaders import AbstractDataLoader
from src.experiment.reporters import ReporterList
from src.trainers.regression import TimeSeriesRegressionTrainer
from src.utils import types
from src.utils.exceptions import StopSweep
from src.utils.iterables import collect_keys_with_prefix


class SweepRunner:
    def __init__(
        self,
        sweep_config: dict[str, Any],
        model_from_parameters: types.TorchParameterizedModel,
        data_loader: AbstractDataLoader,
        reporters: ReporterList,
        print_fn: types.PrintFunction = print,
    ):
        self.config = sweep_config
        self.model_from_parameters = model_from_parameters
        self.data_loader = data_loader
        self.reporters = reporters
        self.print_fn = print_fn

    @property
    def default_model_store_path(self) -> Path:
        return Path("model.pt")

    def build_model(self, parameters: dict):
        """Build model from parameters chosen by sweep agent from available configuration list"""
        model = self.model_from_parameters(parameters)
        self.print_fn("Created model with parameters...")
        return model

    def build_optimizer(self, model, parameters: dict):
        """
        Build optimizer from parameters

        Optimizer parameters are collected by prefix `optimizer_`, since WANDB does not support nested dicts
        and sweeping through optimizer parameters is sometimes required
        """
        optimizer_parameters = collect_keys_with_prefix(parameters, prefix="optimizer_")
        optimizer = config.build_optimizer(model, name=parameters["optimizer"], parameters=optimizer_parameters)
        self.print_fn("Created optimizer...")
        return optimizer

    def build_loss_fn(self, parameters: dict):
        """Build loss function, handles loss parameters using prefix `loss_fn_`, similarly to `build_optimizer`"""
        loss_fn_params = collect_keys_with_prefix(parameters, prefix="loss_fn_")
        loss = config.build_loss_function(name=parameters["loss_function"], parameters=loss_fn_params)
        self.print_fn("Created loss function...")
        return loss

    def build_callback_handler(self):
        """
        Creates callback handled from parameters
        Parameters are given directly in `sweep_config` dict, since they are not swept
        """
        callback_handler = config.build_callback_handler(
            names=self.config["callback_parameters"]["names"],
            parameters=self.config["callback_parameters"]["parameters"],
        )

        self.print_fn("Created callbacks...")
        return callback_handler

    def build_checkpoint_handler(self):
        """
        Creates checkpoint handled from parameters
        Parameters are given directly in `sweep_config` dict, since they are not swept
        """
        checkpoint_list = config.build_checkpoint_list(
            names=self.config["checkpoint_parameters"]["names"],
            parameters=self.config["checkpoint_parameters"]["parameters"],
            restore_from=self.config["checkpoint_parameters"]["restore_from"],
        )
        self.print_fn("Created checkpoints...")
        return checkpoint_list

    def log_trained_model(self, model) -> None:
        """Log trained model as a file to WANDB interface"""
        path = Path(wandb.run.dir) / self.default_model_store_path
        torch.save(model, path)
        self.print_fn(f"Logging model from {path} to WANDB!")
        wandb.save(str(path))

    def run_experiment(self):
        with wandb.init():
            try:
                parameters = wandb.config

                model = self.build_model(parameters)
                optimizer = self.build_optimizer(model, parameters)
                loss = self.build_loss_fn(parameters)
                callback_handler = self.build_callback_handler()
                checkpoint_handler = self.build_checkpoint_handler()

                trainer = TimeSeriesRegressionTrainer(
                    model=model,
                    optimizer=optimizer,
                    loss_function=loss,
                    callback_handler=callback_handler,
                    checkpoint_handler=checkpoint_handler,
                    name=wandb.run.name,
                    n_epochs=parameters["n_epochs"],
                    device=self.config["device"],
                )

                self.print_fn("Starting training...")
                trainer.train(
                    data_loader=self.data_loader.get_training_data(),
                    validation_data_loader=self.data_loader.get_test_data(),
                )
                trainer.post_train()

                self.print_fn("Starting predicting...")
                targets, predictions = trainer.predict(self.data_loader.get_test_data())

                self.print_fn("Creating report...")
                self.reporters(
                    **{
                        "model": model,
                        "targets": targets,
                        "predictions": predictions,
                        "training_summary": trainer.training_summary,
                    }
                )

                self.log_trained_model(model)
                self.print_fn("Run finished!")

            except Exception as e:
                self.print_fn(e)
                self.print_fn(traceback.format_exc())
                raise StopSweep from e

    def run_sweep(self, project_name: str, n_processes: int = 1, n_runs: int = 1):
        """Runs a sweep of experiments with the given config"""
        sweep_id = wandb.sweep(self.config["sweep_config"])
        # TODO: Add multiprocessing starting multiple agents with single sweep_id
        wandb.agent(sweep_id, function=self.run_experiment, count=n_runs, project=project_name)
