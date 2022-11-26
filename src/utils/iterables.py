from typing import Any, Optional


def filter_dict(function: Optional[callable], d: dict) -> dict:
    """Filters dict based on values"""

    def default_function(value: Any) -> bool:
        return value is not None

    if function is None:
        function = default_function

    return {key: value for key, value in d.items() if function(value)}
