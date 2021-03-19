import numpy as np

from arc.consts import arc_cmap


def plot_grid(plot_handle, grid: np.ndarray):
    plot_handle.pcolormesh(
        grid,
        cmap=arc_cmap,
        rasterized=True,
        vmin=0,
        vmax=9,
    )
    plot_handle.set_xticks(np.arange(0, grid.shape[1], 1))
    plot_handle.set_yticks(np.arange(0, grid.shape[0], 1))
    plot_handle.grid()
    plot_handle.set_aspect(1)
