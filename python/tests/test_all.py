import pytest
import compression_rs


def test_sum_as_string():
    assert compression_rs.sum_as_string(1, 1) == "2"
