"""
For building an agent that can find and fix bugs in code.
This file has simple functions that the agent will review and improve.
"""
from __future__ import annotations

import math
from typing import Iterable


def calculate_average(numbers: Iterable[int | float]) -> float:
    """Return the arithmetic mean of *numbers*.

    Args:
        numbers: An iterable of real (int or float) values. Generators accepted.

    Returns:
        The arithmetic mean as a float.

    Raises:
        TypeError:  *numbers* is None, not iterable, or contains bool /
                    non-numeric elements.
        ValueError: *numbers* is empty, or contains NaN or infinite values.
    """
    if numbers is None:
        raise TypeError("None is not a valid input; expected an iterable of numbers.")

    # Materialise generators so we can validate, count, and sum in one pass.
    try:
        items = list(numbers)
    except TypeError:
        raise TypeError(
            f"Expected an iterable of numbers, got {type(numbers).__name__!r}."
        )

    if len(items) == 0:
        raise ValueError("Cannot calculate average of an empty sequence.")

    total = 0.0
    for i, num in enumerate(items):
        # bool is a subclass of int in Python — reject it explicitly.
        if isinstance(num, bool):
            raise TypeError(
                f"Element at index {i} is bool ({num!r}); "
                "only int and float values are accepted."
            )
        if not isinstance(num, (int, float)):
            raise TypeError(
                f"Element at index {i} is {type(num).__name__!r} ({num!r}); "
                "only int and float values are accepted."
            )
        if math.isnan(num):
            raise ValueError(
                f"Element at index {i} contains NaN, which is not a valid number."
            )
        if math.isinf(num):
            raise ValueError(
                f"Element at index {i} is infinite, which is not a valid number."
            )
        total += num

    return total / len(items)


def get_user_name(user: dict) -> str:
    """Return the upper-cased name stored in *user*.

    Args:
        user: A dict that must contain a non-empty string under the key ``"name"``.

    Returns:
        The user's name converted to upper case.

    Raises:
        TypeError:   *user* is None, not a dict, or ``user["name"]`` is not a str.
        KeyError:    ``"name"`` key is absent from *user*.
        ValueError:  ``user["name"]`` is an empty or whitespace-only string.
    """
    if user is None:
        raise TypeError("None is not a valid user; expected a dict.")
    if not isinstance(user, dict):
        raise TypeError(
            f"Expected a dict for 'user', got {type(user).__name__!r}."
        )

    name = user["name"]  # raises KeyError naturally if the key is missing

    if not isinstance(name, str):
        raise TypeError(
            f"'name' must be a string, got {type(name).__name__!r} ({name!r})."
        )
    if not name.strip():
        raise ValueError(
            f"'name' must not be empty or whitespace-only, got {name!r}."
        )

    return name.upper()