import numpy as np
from matplotlib import pyplot as plt

from arc import train_problems, validation_problems, ArcProblem, plot_grid
from arc.types import verify_is_arc_grid

train_problems : list[ArcProblem]

# get a sample problem
prob : ArcProblem = train_problems[0]
io_pair = prob.train_pairs[0]

# get a grid from problem
grid: np.ndarray = io_pair.x  # also has .y

# only an integer array with size 1 to 30 and values 0 to 9 are valid.
# `verify_is_arc_grid` raises AssertionError if one of these conditions don't match.
verify_is_arc_grid(grid)

#visualize
plot_grid(grid)
plt.show()
