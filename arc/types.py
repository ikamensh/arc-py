from typing import List

import numpy as np
from matplotlib import pyplot as plt

from arc.plot import plot_grid

# 2D ndarray of integers, both dimensions from 1 to 30, values from 0 to 9 (integers)
ArcGrid = np.ndarray


def verify_is_arc_grid(grid: np.ndarray):
    assert (
        len(grid.shape) == 2
    ), f"Expected two-dimensional array, got array with shape {grid.shape}."
    assert issubclass(grid.dtype.type, np.integer), f"Expected an integer array."
    assert 0 <= np.amin(grid) <= np.amax(grid) <= 9, (
        f"ARC grid can only contain values between 0 and 9. "
        f"Found interval: [{np.amin(grid)}, {np.amax(grid)}]"
    )


# on ARC tasks, we are allowed to make multiple predictions,
# aiming for at least one of them to be correct.
ArcPrediction = List[ArcGrid]


class ArcIOPair:
    def __init__(self, x: ArcGrid, y: ArcGrid):
        self.x = x
        self.y = y

    def plot(self, show=True, title:str=None, fig_axes = None):

        fig, (ax1, ax2) = fig_axes or plt.subplots(1, 2)
        if title:
            fig.suptitle(title, fontsize=16)
        plot_grid(ax1, self.x)
        ax1.set_title("Input")
        plot_grid(ax2, self.y)
        ax2.set_title("Output")

        if show:
            plt.show()

    def __eq__(self, other):
        if not isinstance(other, ArcIOPair):
            return False
        return self.x == other.x and self.y == other.y


class ArcProblem:
    """A problem in ARC Challenge.

    ARC problem has uid (name of original json file),
    a list of demonstration pairs and a list of test pairs.
    """

    def __init__(
        self, uid: str, demo_pairs: List[ArcIOPair], test_pairs: List[ArcIOPair]
    ):
        self.uid = uid
        self.train_pairs = demo_pairs
        self.test_pairs = test_pairs

    @property
    def test_inputs(self):
        return [p.x for p in self.test_pairs]

    @property
    def test_outputs(self):
        return [p.y for p in self.test_pairs]


    def __repr__(self):
        return f"ArcProblem # {self.uid}"
