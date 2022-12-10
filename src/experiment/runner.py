import traceback
from pathlib import Path
from typing import Any

import wandb
import yaml

from src.experiment.config import build_callbacks, build_loss_function, build_optimizer
from src.experiment.data_loaders import AbstractDataLoader
from src.trainers.regression import TimeSeriesRegressionTrainer
from src.utils import types
from src.utils.exceptions import StopSweep


class SweepRunner:
    def __init__(
        self,
        config: dict[str, Any],
        model_from_parameters: types.TorchParameterizedModel,
        data_loader: AbstractDataLoader,
        reporter: types.TorchReportFunction,
        print_fn: types.PrintFunction = print,
    ):
        self.config = config
        self.model_from_parameters = model_from_parameters
        self.data_loader = data_loader
        self.reporter = reporter
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

                optimizer = build_optimizer(model, parameters["optimizer"], parameters.get("optimizer_parameters", {}))
                self.print_fn("Created optimizer...")

                loss = build_loss_function(parameters["loss_function"], parameters.get("loss_function_parameters", {}))
                self.print_fn("Created loss function...")

                callbacks = build_callbacks(
                    self.config["callback_parameters"]["names"], self.config["callback_parameters"]["parameters"]
                )
                self.print_fn("Created callbacks...")

                trainer = TimeSeriesRegressionTrainer(
                    model=model,
                    optimizer=optimizer,
                    loss_function=loss,
                    callbacks=callbacks,
                    n_epochs=parameters["n_epochs"],
                    device=self.config["device"],
                )

                self.print_fn("Starting training...")
                model = trainer.train(
                    self.data_loader.get_training_data(),
                    validation_data_loader=self.data_loader.get_test_data()
                )

                self.print_fn("Starting predicting...")
                targets, predictions = trainer.predict(self.data_loader.get_test_data())

                self.print_fn("Creating report...")
                metrics = self.reporter(
                    **{
                        "model": model,
                        "targets": targets,
                        "predictions": predictions,
                        "training_summary": trainer.training_summary,
                    }
                )
                wandb.log(metrics)
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
