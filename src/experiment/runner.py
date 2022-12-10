import traceback
from pathlib import Path
from typing import Any

import torch
import wandb
import yaml

from src.experiment.config import build_callbacks, build_checkpoints, build_loss_function, build_optimizer
from src.experiment.data_loaders import AbstractDataLoader
from src.experiment.reporters import ReporterList
from src.trainers.regression import TimeSeriesRegressionTrainer
from src.utils import types
from src.utils.exceptions import StopSweep
from src.utils.iterables import collect_keys_with_prefix


class SweepRunner:
    def __init__(
        self,
        config: dict[str, Any],
        model_from_parameters: types.TorchParameterizedModel,
        data_loader: AbstractDataLoader,
        reporters: ReporterList,
        print_fn: types.PrintFunction = print,
    ):
        self.config = config
        self.model_from_parameters = model_from_parameters
        self.data_loader = data_loader
        self.reporters = reporters
        self.print_fn = print_fn

    @classmethod
    def from_yaml_config(
        cls,
        config_path: Path,
        model_from_parameters: types.TorchParameterizedModel,
        data_loader: AbstractDataLoader,
        report_fn: types.TorchReportFunction,
        print_fn: types.PrintFunction = print,
    ):
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        return cls(config, model_from_parameters, data_loader, report_fn, print_fn)

    def run_experiment(self):
        with wandb.init():
            try:
                parameters = wandb.config

                model = self.model_from_parameters(parameters)
                self.print_fn("Created model with parameters...")

                optimizer_parameters = collect_keys_with_prefix(parameters, prefix="optimizer_")
                optimizer = build_optimizer(model, name=parameters["optimizer"], parameters=optimizer_parameters)
                self.print_fn("Created optimizer...")

                loss_fn_params = collect_keys_with_prefix(parameters, prefix="loss_fn_")
                loss = build_loss_function(name=parameters["loss_function"], parameters=loss_fn_params)
                self.print_fn("Created loss function...")

                callbacks = build_callbacks(
                    self.config["callback_parameters"]["names"], self.config["callback_parameters"]["parameters"]
                )
                self.print_fn("Created callbacks...")

                checkpoints = build_checkpoints(
                    self.config["checkpoint_parameters"]["names"],
                    self.config["checkpoint_parameters"]["parameters"],
                    self.config["checkpoint_parameters"]["restore_from"],
                )
                self.print_fn("Created checkpoints...")

                trainer = TimeSeriesRegressionTrainer(
                    model=model,
                    optimizer=optimizer,
                    loss_function=loss,
                    callbacks=callbacks,
                    checkpoints=checkpoints,
                    name=wandb.run.name,
                    n_epochs=parameters["n_epochs"],
                    device=self.config["device"],
                )

                self.print_fn("Starting training...")
                trainer.train(
                    self.data_loader.get_training_data(), validation_data_loader=self.data_loader.get_test_data()
                )
                model = trainer.post_train()

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

                path = Path(wandb.run.dir) / "model.pt"
                torch.save(model, path)
                self.print_fn(f"Logging model from {path} to WANDB!")
                wandb.save(str(path))

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
