"""Comprehensive pytest tests for utils.py."""

from __future__ import annotations

import math

import pytest

from utils import calculate_average, get_user_name


# ===========================================================================
# calculate_average – happy-path tests
# ===========================================================================

class TestCalculateAverageHappyPath:
    def test_integers(self):
        assert calculate_average([1, 2, 3]) == 2.0

    def test_floats(self):
        assert math.isclose(calculate_average([1.5, 2.5, 3.0]), 7.0 / 3)

    def test_mixed_int_float(self):
        assert calculate_average([1, 2.0, 3]) == pytest.approx(2.0)

    def test_single_element(self):
        assert calculate_average([42]) == 42.0

    def test_negative_numbers(self):
        assert calculate_average([-1, -2, -3]) == pytest.approx(-2.0)

    def test_zeros(self):
        assert calculate_average([0, 0, 0]) == 0.0

    def test_large_list(self):
        nums = list(range(1, 101))  # 1..100  → average = 50.5
        assert calculate_average(nums) == pytest.approx(50.5)

    def test_tuple_input(self):
        """Any iterable should work."""
        assert calculate_average((10, 20, 30)) == 20.0

    def test_generator_input(self):
        assert calculate_average(x for x in [4, 6]) == 5.0

    def test_returns_float(self):
        result = calculate_average([4, 8])
        assert isinstance(result, float)


# ===========================================================================
# calculate_average – error / edge-case tests
# ===========================================================================

class TestCalculateAverageErrors:
    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError, match="None"):
            calculate_average(None)

    def test_empty_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_average([])

    def test_string_raises_typeerror(self):
        """A bare string is iterable but should be rejected."""
        with pytest.raises(TypeError):
            calculate_average("123")

    def test_non_iterable_int_raises_typeerror(self):
        with pytest.raises(TypeError):
            calculate_average(42)

    def test_non_numeric_element_raises_typeerror(self):
        with pytest.raises(TypeError, match="index 1"):
            calculate_average([1, "two", 3])

    def test_none_element_raises_typeerror(self):
        with pytest.raises(TypeError):
            calculate_average([1, None, 3])

    def test_bool_element_raises_typeerror(self):
        """bool is a subclass of int; it should be explicitly rejected."""
        with pytest.raises(TypeError, match="bool"):
            calculate_average([1, True, 3])

    def test_mixed_valid_invalid_raises_typeerror(self):
        with pytest.raises(TypeError):
            calculate_average([1.0, 2.0, "oops"])

    def test_list_of_strings_raises_typeerror(self):
        with pytest.raises(TypeError):
            calculate_average(["a", "b", "c"])

    def test_dict_raises_typeerror(self):
        """Iterating a dict yields keys; should still fail on non-numeric."""
        with pytest.raises(TypeError):
            calculate_average({"a": 1, "b": 2})

    def test_nan_element_raises_valueerror(self):
        """NaN is a float but must be rejected to avoid a silent NaN mean."""
        with pytest.raises(ValueError, match="NaN"):
            calculate_average([1, 2, float("nan")])

    def test_inf_element_raises_valueerror(self):
        """Positive infinity must be rejected."""
        with pytest.raises(ValueError, match="infinite"):
            calculate_average([1, 2, float("inf")])

    def test_neg_inf_element_raises_valueerror(self):
        """Negative infinity must be rejected."""
        with pytest.raises(ValueError, match="infinite"):
            calculate_average([1, 2, float("-inf")])


# ===========================================================================
# get_user_name – happy-path tests
# ===========================================================================

class TestGetUserNameHappyPath:
    def test_basic(self):
        assert get_user_name({"name": "alice"}) == "ALICE"

    def test_already_uppercase(self):
        assert get_user_name({"name": "BOB"}) == "BOB"

    def test_mixed_case(self):
        assert get_user_name({"name": "cHaRlIe"}) == "CHARLIE"

    def test_extra_keys_ignored(self):
        assert get_user_name({"name": "dave", "age": 30}) == "DAVE"

    def test_single_character(self):
        assert get_user_name({"name": "x"}) == "X"

    def test_name_with_spaces(self):
        assert get_user_name({"name": "john doe"}) == "JOHN DOE"

    def test_unicode_name(self):
        assert get_user_name({"name": "élodie"}) == "ÉLODIE"

    def test_returns_str(self):
        result = get_user_name({"name": "test"})
        assert isinstance(result, str)


# ===========================================================================
# get_user_name – error / edge-case tests
# ===========================================================================

class TestGetUserNameErrors:
    def test_none_raises_typeerror(self):
        with pytest.raises(TypeError, match="None"):
            get_user_name(None)

    def test_non_dict_string_raises_typeerror(self):
        with pytest.raises(TypeError):
            get_user_name("alice")

    def test_non_dict_list_raises_typeerror(self):
        with pytest.raises(TypeError):
            get_user_name(["alice"])

    def test_non_dict_int_raises_typeerror(self):
        with pytest.raises(TypeError):
            get_user_name(42)

    def test_missing_name_key_raises_keyerror(self):
        with pytest.raises(KeyError, match="name"):
            get_user_name({"username": "alice"})

    def test_empty_dict_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_user_name({})

    def test_name_is_none_raises_typeerror(self):
        with pytest.raises(TypeError):
            get_user_name({"name": None})

    def test_name_is_int_raises_typeerror(self):
        with pytest.raises(TypeError):
            get_user_name({"name": 123})

    def test_name_is_list_raises_typeerror(self):
        with pytest.raises(TypeError):
            get_user_name({"name": ["alice"]})

    def test_empty_string_name_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            get_user_name({"name": ""})

    def test_whitespace_only_name_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            get_user_name({"name": "   "})

    def test_tab_only_name_raises_valueerror(self):
        with pytest.raises(ValueError):
            get_user_name({"name": "\t\n"})
