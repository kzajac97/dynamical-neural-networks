import pytest
import torch

from src.models.linear import TimeSeriesLinear


@pytest.mark.parametrize(
    "batch, n_input_time_steps, n_output_time_steps, n_input_state_variables, n_output_state_variables, expected_shape",
    [
        (1, 100, 100, 1, 1, (1, 100, 1)),
        (32, 100, 100, 1, 1, (32, 100, 1)),
        (16, 100, 100, 3, 3, (16, 100, 3)),
        (12, 10, 10, 5, 1, (12, 10, 1)),
        (16, 10, 10, 1, 5, (16, 10, 5)),
        (1, 50, 10, 3, 3, (1, 10, 3)),
        (1, 50, 25, 1, 1, (1, 25, 1)),
        (1, 1, 1, 1, 1, (1, 1, 1)),
    ],
)
def test_time_series_linear(
    batch: int,
    n_input_time_steps: int,
    n_output_time_steps: int,
    n_input_state_variables: int,
    n_output_state_variables: int,
    expected_shape: tuple,
):
    """Tests if TimeSeriesLiner module produces expected tensor shapes"""
    inputs = torch.randn(batch, n_input_time_steps, n_input_state_variables)

    layer = TimeSeriesLinear(
        n_input_time_steps=n_input_time_steps,
        n_output_time_steps=n_output_time_steps,
        n_input_state_variables=n_input_state_variables,
        n_output_state_variables=n_output_state_variables,
    )

    with torch.no_grad():
        outputs = layer(inputs)

    assert outputs.numpy().shape == expected_shape
