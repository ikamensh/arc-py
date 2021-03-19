from typing import Optional, List

import numpy as np
from matplotlib import pyplot as plt

from arc.plot import plot_grid


class ArcIOPair:
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray]):
        self.x = x
        self.y = y

    def plot(self, show=True):
        if self.y is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plot_grid(ax1, self.x)
            plot_grid(ax2, self.y)
        else:
            plot_grid(plt, self.x)

        if show:
            plt.show()

    def __eq__(self, other):
        if not isinstance(other, ArcIOPair):
            return False
        return self.x == other.x and self.y == other.y


class ArcProblem:
    def __init__(self, train_pairs: List[ArcIOPair], test_pairs: List[ArcIOPair]):
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
