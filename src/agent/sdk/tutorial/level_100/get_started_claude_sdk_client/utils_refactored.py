"""Utility helpers used across the project."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Union


# ---------------------------------------------------------------------------
# calculate_average
# ---------------------------------------------------------------------------

def calculate_average(numbers: Iterable[Union[int, float]]) -> float:
    """Return the arithmetic mean of *numbers*.

    Args:
        numbers: An iterable of numeric values (``int`` or ``float``).
                 Strings and other non-numeric types are rejected even if
                 they look like numbers.

    Returns:
        The arithmetic mean as a ``float``.

    Raises:
        TypeError:  If *numbers* is ``None`` or not iterable, or if any
                    element is not an ``int`` or ``float`` (``bool`` values
                    are also rejected even though they are a subclass of
                    ``int``).
        ValueError: If *numbers* is empty, or if any element is ``NaN`` or
                    infinite (``float('nan')``, ``float('inf')``,
                    ``float('-inf')``).  These IEEE 754 special values would
                    silently corrupt the computed mean.

    Examples:
        >>> calculate_average([1, 2, 3])
        2.0
        >>> calculate_average([])
        Traceback (most recent call last):
            ...
        ValueError: numbers must not be empty
    """
    if numbers is None:
        raise TypeError("numbers must not be None")
    if isinstance(numbers, str):
        # A string is technically iterable but clearly wrong here.
        raise TypeError(
            f"numbers must be an iterable of numeric values, got str"
        )
    if not isinstance(numbers, Iterable):
        raise TypeError(
            f"numbers must be an iterable, got {type(numbers).__name__}"
        )

    validated: list[Union[int, float]] = []
    for i, num in enumerate(numbers):
        # Check bool first: bool is a subclass of int so the isinstance guard
        # below would accept True/False silently without this early exit.
        if isinstance(num, bool):
            raise TypeError(
                f"All elements must be int or float; "
                f"element at index {i} is bool: {num!r}"
            )
        if not isinstance(num, (int, float)):
            raise TypeError(
                f"All elements must be int or float; "
                f"element at index {i} is {type(num).__name__!r}: {num!r}"
            )
        # Reject IEEE 754 special values that would silently corrupt the mean.
        if isinstance(num, float) and math.isnan(num):
            raise ValueError(
                f"All elements must be finite numbers; "
                f"element at index {i} is NaN"
            )
        if isinstance(num, float) and math.isinf(num):
            raise ValueError(
                f"All elements must be finite numbers; "
                f"element at index {i} is infinite: {num!r}"
            )
        validated.append(num)

    if not validated:
        raise ValueError("numbers must not be empty")

    # math.fsum gives exact floating-point summation (no accumulated error).
    return math.fsum(validated) / len(validated)


# ---------------------------------------------------------------------------
# get_user_name
# ---------------------------------------------------------------------------

def get_user_name(user: dict) -> str:
    """Return the uppercased name from a *user* mapping.

    Args:
        user: A ``dict`` that must contain a ``"name"`` key whose value is
              a non-empty ``str``.

    Returns:
        The uppercased user name.

    Raises:
        TypeError:  If *user* is ``None``, not a ``dict``, or if
                    ``user["name"]`` is not a ``str``.
        KeyError:   If *user* does not contain the ``"name"`` key.
        ValueError: If ``user["name"]`` is an empty or whitespace-only string.

    Examples:
        >>> get_user_name({"name": "alice"})
        'ALICE'
    """
    if user is None:
        raise TypeError("user must not be None")
    if not isinstance(user, dict):
        raise TypeError(
            f"user must be a dict, got {type(user).__name__!r}"
        )
    if "name" not in user:
        raise KeyError("user dict must contain a 'name' key")

    name = user["name"]
    if name is None:
        raise TypeError("user['name'] must not be None")
    if not isinstance(name, str):
        raise TypeError(
            f"user['name'] must be a str, got {type(name).__name__!r}: {name!r}"
        )
    if not name.strip():
        raise ValueError("user['name'] must not be empty or whitespace-only")

    return name.upper()
