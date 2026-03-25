"""Light/dark theme for blog plots, matching the site's warm paper aesthetic.

Usage:
    from plot_theme import savefig

    def make_plot(t):
        fig, ax = plt.subplots()
        ax.plot(x, y, color=t.red)
        ax.plot(x, z, color=t.blue)
        return fig

    savefig(make_plot, 'output.png')

Theme attributes:
    bg, fg, fg_light, fg_faded     — background and text hierarchy
    accent, rule, grid, grid_minor  — chrome colors
    legend_edge, legend_alpha       — legend styling

    red, blue, green, purple,       — data colors tuned per theme
    amber, gray, cyan               — (warm on light, lifted on dark)

    cycle                           — default color cycle list
"""
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

_DEFAULTS = None


class Theme:
    __slots__ = [
        'name', 'bg', 'fg', 'fg_light', 'fg_faded', 'accent',
        'rule', 'grid', 'grid_minor', 'legend_edge', 'legend_alpha',
        'red', 'blue', 'green', 'purple', 'amber', 'gray', 'cyan',
        'cycle',
    ]

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def apply(self):
        matplotlib.rcParams.update({
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.facecolor': self.bg,
            'axes.facecolor': self.bg,
            'savefig.facecolor': self.bg,
            'text.color': self.fg,
            'axes.edgecolor': self.fg_light,
            'axes.labelcolor': self.fg,
            'xtick.color': self.fg_light,
            'ytick.color': self.fg_light,
            'grid.color': self.grid,
            'axes.prop_cycle': matplotlib.cycler(color=self.cycle),
            'legend.edgecolor': self.legend_edge,
            'legend.framealpha': self.legend_alpha,
            'legend.fancybox': False,
        })


# ---------------------------------------------------------------------------
# Light — warm paper (#faf9f6), earthy muted data colors
# ---------------------------------------------------------------------------
LIGHT = Theme(
    name='light',
    bg='#f8f6f1',
    fg='#3a352d',
    fg_light='#6b6560',
    fg_faded='#938e86',
    accent='#8b6b4a',
    rule='#ddd9d0',
    grid='#d5d0c8',
    grid_minor='#e8e4dc',
    legend_edge='#d5d0c8',
    legend_alpha=0.95,
    red='#b5503a',        # burnt sienna
    blue='#3d7ea0',       # warm steel
    green='#6a8a55',      # olive sage
    purple='#905a8a',     # warm plum
    amber='#b08030',      # warm gold
    gray='#7a7068',       # warm neutral
    cyan='#2a8a8a',       # teal
    cycle=['#3d7ea0', '#b5503a', '#6a8a55', '#905a8a', '#b08030', '#2a8a8a', '#7a7068'],
)

# ---------------------------------------------------------------------------
# Dark — deep warm black (#1c1a17), lifted/softer data colors
# ---------------------------------------------------------------------------
DARK = Theme(
    name='dark',
    bg='#201e1a',
    fg='#d8d0c4',
    fg_light='#a09888',
    fg_faded='#78726a',
    accent='#c4a07a',
    rule='#3d3830',
    grid='#38332c',
    grid_minor='#2e2a24',
    legend_edge='#3d3830',
    legend_alpha=0.95,
    red='#cc7060',        # earthy brick
    blue='#6ab0d0',       # sky steel
    green='#90c480',      # warm lifted sage
    purple='#b890c0',     # soft lavender
    amber='#d8b060',      # warm gold
    gray='#b0a8a0',       # warm neutral
    cyan='#50b8b8',       # teal
    cycle=['#6ab0d0', '#cc7060', '#90c480', '#b890c0', '#d8b060', '#50b8b8', '#b0a8a0'],
)

THEMES = [LIGHT, DARK]


def savefig(plot_fn, path, **kwargs):
    """Call plot_fn(theme) for each theme, saving path and path_dark.png."""
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = matplotlib.rcParams.copy()
    path = Path(path)
    kwargs.setdefault('dpi', 200)
    kwargs.setdefault('bbox_inches', 'tight')
    for theme in THEMES:
        matplotlib.rcParams.update(_DEFAULTS)
        theme.apply()
        fig = plot_fn(theme)
        suffix = '' if theme is LIGHT else '_dark'
        out = path.with_stem(path.stem + suffix)
        fig.savefig(out, **kwargs)
        plt.close(fig)
        print(f'Saved {out}')
    matplotlib.rcParams.update(_DEFAULTS)
