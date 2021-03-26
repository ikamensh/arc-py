import numpy as np

from arc import get_train_problem_by_uid
from arc import ArcColors as ac


drop_prob = get_train_problem_by_uid("54d82841")


def test_axes():
    """expects to have origin in upper-left corner, axis_0 to go down, axis_1 to go right.

    See the drop problem in ARC web interface for reference."""

    has_drops = drop_prob.train_pairs[0].y

    assert np.all(has_drops[-1] == [0, 0, ac.YELLOW, 0, 0, 0, ac.YELLOW, 0])
