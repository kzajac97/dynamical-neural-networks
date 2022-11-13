import pytest

from src.experiment.callbacks import EarlyStoppingCallback


@pytest.mark.parametrize(
    "patience, expected_metrics",
    [
        (10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (5, [5, 6, 7, 8, 9]),
        (3, [7, 8, 9]),
        (7, [3, 4, 5, 6, 7, 8, 9]),
        (None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ],
)
def test_patience_slicing(patience: int, expected_metrics: list[float]):
    mock_inputs = list(range(10))
    callback = EarlyStoppingCallback(metric_name="MOCK_METRIC", moving_average_window_size=0, patience=patience)

    assert mock_inputs[callback.patience_slice] == expected_metrics
