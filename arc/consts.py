from matplotlib.colors import ListedColormap

# RGB
colors_rgb = {
    0: (0x00, 0x00, 0x00),
    1: (0x00, 0x74, 0xD9),
    2: (0xFF, 0x41, 0x36),
    3: (0x2E, 0xCC, 0x40),
    4: (0xFF, 0xDC, 0x00),
    5: (0xA0, 0xA0, 0xA0),
    6: (0xF0, 0x12, 0xBE),
    7: (0xFF, 0x85, 0x1B),
    8: (0x7F, 0xDB, 0xFF),
    9: (0x87, 0x0C, 0x25),
}

_float_colors = [tuple(c / 255 for c in col) for col in colors_rgb.values()]
arc_cmap = ListedColormap(_float_colors)

class ArcColors:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    FUCHSIA = 6
    ORANGE = 7
    TEAL = 8
    BROWN = 9
