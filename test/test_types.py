import pytest
import numpy as np

from arc.types import verify_is_arc_grid


def test_black_passes():
    background = np.zeros((3,3), np.uint8)

    verify_is_arc_grid(background)
    verify_is_arc_grid(background.astype(np.int64))


def test_3d_fails():
    cube = np.zeros((3, 3, 3), np.uint8)

    with pytest.raises(AssertionError):
        verify_is_arc_grid(cube)


def test_big_values_fail():
    grid = 111 * np.ones((3, 3), np.uint8)

    with pytest.raises(AssertionError):
        verify_is_arc_grid(grid)


def test_float_fail():
    grid =  np.random.normal(size=(3, 3))

    with pytest.raises(AssertionError):
        verify_is_arc_grid(grid)
