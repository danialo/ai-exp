import pytest

def test_passing_test():
    assert True

def test_failing_test():
    assert False

def test_error_test():
    raise Exception('Error')

def test_edge_case():
    assert 1 == 1