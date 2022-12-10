import pytest

from src.utils.iterables import collect_keys_with_prefix, filter_dict


@pytest.mark.parametrize(
    "input_dict, function, expected_dict",
    [
        ({"A": 1, "B": 2, "C": 3}, None, {"A": 1, "B": 2, "C": 3}),
        ({"A": 1, "B": 2, "C": None, "D": None}, None, {"A": 1, "B": 2}),
        (
            {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8},
            lambda x: x % 2 == 0,
            {"B": 2, "D": 4, "F": 6, "H": 8},
        ),
        ({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8}, lambda x: x > 5, {"F": 6, "G": 7, "H": 8}),
    ],
)
def test_filter_dict(input_dict, function, expected_dict):
    result = filter_dict(function, input_dict)
    assert result == expected_dict


@pytest.mark.parametrize(
    "input_dict, prefix, expected_dict",
    [
        ({"key_a": 1, "key_b": 2, "key_c": 3}, "key", {"key_a": 1, "key_b": 2, "key_c": 3}),
        ({"key_a": 1, "b": 2, "key_c": 3}, "key", {"key_a": 1, "key_c": 3}),
        ({"key_a": 1, "key_b": 2, "key_c": 3}, "x", {}),
    ],
)
def test_collect_keys_with_prefix(input_dict, prefix, expected_dict):
    result = collect_keys_with_prefix(input_dict, prefix)
    assert result == expected_dict
