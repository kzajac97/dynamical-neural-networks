from typing import Sequence

import numpy as np
import pandas as pd
from sklearn import metrics

import src.utils.numpy as utils


def regression_report(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    precision: int = 4,
    width: int = 32,
    use_percentage: bool = False,
) -> str:
    """
    Returns detailed regression report as string

    :param y_true: sequence of ground truth values for regression problem
    :param y_pred: sequence of values predicted by the model
    :param precision: keyword only, precision to which metrics are expressed, defaults to 4
    :param width: keyword only, spacing between columns, defaults to 32,
                  too small values might results in unreadable table
    :param use_percentage: if True print normalized values in %

    :return: regression metrics and data statistics as single string
    """
    y_true, y_pred = utils.cast_to_arrays(y_true, y_pred)
    utils.verify_shape(y_true, y_pred)

    precision_marker = "%" if use_percentage else "f"

    report = ""
    # Headers
    report += f"{'':<{width}}{'Absolute':<{width}}{'Normalized':<{width}}\n\n"
    # Metrics
    report += (
        f"{'Mean Squared Error:':<{width}}"
        f"{metrics.mean_squared_error(y_true=y_true, y_pred=y_pred):<{width}.{precision}f}"
        f"{normalized_mean_square_error(y_true=y_true, y_pred=y_pred):<{width}.{precision}{precision_marker}}\n"
    )

    report += (
        f"{'Root Mean Squared Error:':<{width}}"
        f"{metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False):<{width}.{precision}f}"
        f"{normalized_root_mean_square_error(y_true=y_true, y_pred=y_pred):<{width}.{precision}{precision_marker}}\n"
    )

    report += (
        f"{'Mean Absolute Error:':<{width}}"
        f"{metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred):<{width}.{precision}f}"
        f"{normalized_mean_absolute_error(y_true=y_true, y_pred=y_pred):<{width}.{precision}{precision_marker}}\n"
    )

    report += (
        f"{'Median Absolute Error:':<{width}}"
        f"{metrics.median_absolute_error(y_true=y_true, y_pred=y_pred):<{width}.{precision}f}"
        f"{normalized_median_absolute_error(y_true=y_true, y_pred=y_pred):<{width}.{precision}{precision_marker}}\n"
    )

    report += (
        f"{'Max Error:':<{width}}"
        f"{metrics.max_error(y_true=y_true, y_pred=y_pred):<{width}.{precision}f}"
        f"{normalized_max_error(y_true=y_true, y_pred=y_pred):<{width}.{precision}{precision_marker}}\n"
    )

    report += f"{'R2':<{width}}{'':<{width}}{metrics.r2_score(y_true=y_true, y_pred=y_pred):<{width}.{precision}f}\n"
    report += "\n\n"
    # Statistics
    report += f"{'':<{width}}{'True':<{width}}{'Predicted':<{width}}\n\n"
    report += f"{'Mean:':<{width}}{np.mean(y_true):<{width}.{precision}f}{np.mean(y_pred):<{width}.{precision}f}\n"
    report += f"{'std:':<{width}}{np.std(y_true):<{width}.{precision}f}{np.std(y_pred):<{width}.{precision}f}\n"

    return report


def normalized_mean_square_error(y_true: Sequence, y_pred: Sequence) -> np.array:
    """
    Computes MSE normalized by magnitudes of true values
    It expresses how much the prediction was off relatively to the true values
    Values over 1 are possible, meaning that model was off by more than 100%

    :param y_true: sequence of ground truth values for regression problem
    :param y_pred: sequence of values predicted by the model

    :return: MSE expressed as fraction of ground truth values
    """
    y_true, y_pred = utils.cast_to_arrays(y_true, y_pred)
    utils.verify_shape(y_true, y_pred)

    return np.mean(np.power((y_true - y_pred), 2)) / np.std(y_true)


def normalized_root_mean_square_error(y_true: Sequence, y_pred: Sequence) -> np.array:
    """
    Computes RMSE normalized by magnitudes of true values

    :param y_true: sequence of ground truth values for regression problem
    :param y_pred: sequence of values predicted by the model

    :return: RMSE expressed as fraction of ground truth values
    """
    y_true, y_pred = utils.cast_to_arrays(y_true, y_pred)
    utils.verify_shape(y_true, y_pred)

    return np.sqrt(np.mean(np.power((y_true - y_pred), 2))) / np.std(y_true)


def normalized_mean_absolute_error(y_true: Sequence, y_pred: Sequence) -> np.array:
    """
    Computes MAE normalized by magnitudes of true values

    :param y_true: sequence of ground truth values for regression problem
    :param y_pred: sequence of values predicted by the model

    :return: MAE expressed as fraction of ground truth values
    """
    y_true, y_pred = utils.cast_to_arrays(y_true, y_pred)
    utils.verify_shape(y_true, y_pred)

    return np.mean(np.abs(y_true - y_pred)) / np.std(y_true)


def normalized_median_absolute_error(y_true: Sequence, y_pred: Sequence) -> np.array:
    """
    Computes median absolute error normalized by magnitudes of true values
    It expresses the most common prediction error relatively to the true value

    :param y_true: sequence of ground truth values for regression problem
    :param y_pred: sequence of values predicted by the model

    :return: Median absolute error expressed as fraction of ground truth values
    """
    y_true, y_pred = utils.cast_to_arrays(y_true, y_pred)
    utils.verify_shape(y_true, y_pred)

    return np.median(np.abs(y_true - y_pred)) / np.std(y_true)


def normalized_max_error(y_true: Sequence, y_pred: Sequence) -> np.array:
    """
    Computes max error normalised by magnitude of sample causing it

    :param y_true: sequence of ground truth values for regression problem
    :param y_pred: sequence of values predicted by the model

    :return: Max error expressed as fraction of ground truth value causing this error
    """
    y_true, y_pred = utils.cast_to_arrays(y_true, y_pred)
    utils.verify_shape(y_true, y_pred)

    error = np.abs(y_true - y_pred)
    max_error_id = np.argmax(error)

    return np.abs(error[max_error_id]) / np.std(y_true)


def regression_score(y_true: np.array, y_pred: np.array) -> pd.Series:
    """
    The same output as regression_report stored in pandas series for
    efficiently writing it to file or computing stats using pandas DataFrame

    :param y_true: sequence of ground truth values for regression problem
    :param y_pred: sequence of values predicted by the model

    :return: pandas Series with all relevant metrics for regression problems
    """
    return pd.Series(
        [
            metrics.mean_squared_error(y_true=y_true, y_pred=y_pred),
            normalized_mean_square_error(y_true=y_true, y_pred=y_pred),
            metrics.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False),
            normalized_root_mean_square_error(y_true=y_true, y_pred=y_pred),
            metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred),
            normalized_mean_absolute_error(y_true=y_true, y_pred=y_pred),
            metrics.median_absolute_error(y_true=y_true, y_pred=y_pred),
            normalized_median_absolute_error(y_true=y_true, y_pred=y_pred),
            metrics.max_error(y_true=y_true, y_pred=y_pred),
            metrics.max_error(y_true=y_true, y_pred=y_pred) / np.max(y_true),
            metrics.r2_score(y_true=y_true, y_pred=y_pred),
            np.mean(y_true),
            np.std(y_true),
            np.mean(y_pred),
            np.std(y_pred),
        ],
        index=[
            "MSE",
            "NMSE",
            "RMSE",
            "NRMSE",
            "MAE",
            "NMAE",
            "MDE",
            "NMDE",
            "MAX",
            "NMAX",
            "R2",
            "True Mean",
            "True Std",
            "Predicted Mean",
            "Predicted Std",
        ],
    )
