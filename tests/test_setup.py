"""Test to validate basic project setup."""


def test_imports():
    """Test that core dependencies can be imported."""
    import fastapi
    import pydantic
    import sqlmodel
    import chromadb
    import numpy

    assert fastapi is not None
    assert pydantic is not None
    assert sqlmodel is not None
    assert chromadb is not None
    assert numpy is not None


def test_placeholder():
    """Placeholder test to verify pytest works."""
    assert True
