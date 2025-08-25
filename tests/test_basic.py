"""
Basic test cases
"""
import pytest


def test_example():
    """Example test case"""
    assert True


def test_addition():
    """Test basic addition"""
    assert 2 + 2 == 4


if __name__ == "__main__":
    pytest.main([__file__])
