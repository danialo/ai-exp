import pytest

from module_under_test import some_function

def test_some_function_valid_cases():
    \"\"\"Test valid cases for some_function.\"\"\"
    # Assuming some_function takes an integer and returns its square
    assert some_function(2) == 4
    assert some_function(-2) == 4
    assert some_function(0) == 0

def test_some_function_edge_cases():
    \"\"\"Test edge cases for some_function.\"\"\"
    # Edge case: very large number
    large_num = 10**10
    assert some_function(large_num) == large_num ** 2

    # Edge case: very small number
    small_num = -10**10
    assert some_function(small_num) == small_num ** 2

def test_some_function_type_error():
    \"\"\"Test invalid input types for some_function.\"\"\"
    with pytest.raises(TypeError):
        some_function(\"string\")

    with pytest.raises(TypeError):
        some_function(None)

    with pytest.raises(TypeError):
        some_function(5.5)  # if floats are not allowed

def test_some_function_with_zero():
    \"\"\"Test some_function with zero input.\"\"\"
    assert some_function(0) == 0

def test_some_function_with_negative_numbers():
    \"\"\"Test some_function with negative numbers.\"\"\"
    assert some_function(-3) == 9
    assert some_function(-1) == 1

def test_some_function_with_large_negative_number():
    \"\"\"Test some_function with a large negative number.\"\"\"
    large_negative = -10**12
    assert some_function(large_negative) == large_negative ** 2
