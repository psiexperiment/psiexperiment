'''
Shared look for the icons of psi programs.

Every psi program's icon is the same frame -- a dark blue square with a thick
white border -- around a motif of its own (a chirp for cftscal, noise for
noise-exp, the circuit-board trident for psi itself). Building them all through
`make_icon` keeps the frame, palette and output sizes identical, so the
programs read as one family on the taskbar.

Each program keeps a small `make_icon.py` script that draws its motif and
calls `make_icon`. Rerun that script to regenerate the icon; the resulting
image files are what the program ships, so this module (and matplotlib and
Pillow, which it needs) is only required when making an icon, never when
running the program. Install them with::

    pip install psiexperiment[icons]

Examples
--------
A minimal script that draws a sine wave inside the standard frame::

    import numpy as np
    from psi.launcher.icons import make_icon, plot_signal

    def draw(ax):
        t = np.linspace(0, 1, 100)
        plot_signal(ax, t, np.sin(2 * np.pi * 2 * t))

    make_icon(draw, 'main-icon.png', 'main-icon.ico')
'''
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image


#: Color of the square behind the motif.
BACKGROUND = 'midnightblue'

#: Color of the frame and of the motif's lines.
FOREGROUND = 'white'

#: Color for filled areas of the motif (e.g., the area under a signal).
FILL = 'cornflowerblue'

#: Width of the frame, in points. Half of it falls outside the image, so the
#: visible border is half as wide.
FRAME_WIDTH = 10

#: Width of the motif's lines, in points.
TRACE_WIDTH = 4

#: Width and height of the PNG, in pixels.
PNG_SIZE = 256

#: Sizes Windows picks from when it renders the icon (taskbar, alt-tab,
#: Explorer, ...).
ICO_SIZES = [(s, s) for s in (16, 24, 32, 48, 64, 128, 256)]


def make_icon(draw, png_path, ico_path=None, xlim=(-0.05, 1.05),
              ylim=(-1.5, 1.5)):
    '''
    Draw an icon in the standard frame and save it.

    Parameters
    ----------
    draw : callable
        Called as `draw(ax)` with the matplotlib axes to draw the motif on.
        The axes fill the whole icon and are hidden; use data coordinates
        within `xlim` and `ylim`.
    png_path : str or Path
        Where to save the icon as a `PNG_SIZE` x `PNG_SIZE` PNG. This is the
        file to load as a window icon.
    ico_path : str or Path, optional
        Where to also save a Windows `.ico` holding every size in `ICO_SIZES`
        (e.g., for a desktop shortcut or a frozen executable). Skipped if not
        provided.
    xlim : tuple of float
        Range of data coordinates spanning the icon from left to right.
    ylim : tuple of float
        Range of data coordinates spanning the icon from bottom to top.

    Returns
    -------
    png_path : Path
        Path the PNG was saved to.
    '''
    # A bare Figure rather than pyplot, so making an icon neither needs a GUI
    # backend nor leaves figures open in pyplot's global state.
    fig = Figure(figsize=(1, 1), dpi=PNG_SIZE, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    background = Rectangle([0, 0], width=1, height=1, facecolor=BACKGROUND,
                           edgecolor='none', transform=ax.transAxes)
    ax.add_patch(background)

    # Fix the limits before drawing, and keep plotting calls such as
    # fill_between from rescaling the axes to fit their data, so that `draw`
    # works in the final coordinates.
    ax.axis(xmin=xlim[0], xmax=xlim[1], ymin=ylim[0], ymax=ylim[1])
    ax.set_autoscale_on(False)

    draw(ax)

    # Drawn above the motif so that anything reaching the edge of the icon
    # disappears under the frame rather than being cut off abruptly.
    border = Rectangle([0, 0], width=1, height=1, facecolor='none',
                       edgecolor=FOREGROUND, linewidth=FRAME_WIDTH,
                       transform=ax.transAxes, zorder=3)
    ax.add_patch(border)

    png_path = Path(png_path)
    fig.savefig(png_path, transparent=False, pad_inches=0)

    if ico_path is not None:
        image = Image.open(png_path).convert('RGBA')
        image.save(ico_path, sizes=ICO_SIZES)

    return png_path


def plot_signal(ax, t, y, floor=None, width=TRACE_WIDTH):
    '''
    Draw a signal as a white trace over a filled area.

    This is the motif of the signal-based icons (cftscal's chirp,
    noise-exp's noise): the area below the signal is filled with `FILL` and
    the signal itself is drawn on top in `FOREGROUND`.

    Parameters
    ----------
    ax : matplotlib axes
        The axes passed to the `draw` function of `make_icon`.
    t : array
        Horizontal positions of the samples, in data coordinates.
    y : array
        Signal value at each position, in data coordinates.
    floor : float, optional
        Lower edge of the filled area. Defaults to the bottom of the axes,
        which fills all the way to the frame.
    width : float
        Width of the trace, in points.
    '''
    if floor is None:
        floor = ax.get_ylim()[0]
    ax.fill_between(t, y, floor, color=FILL)
    ax.plot(t, y, color=FOREGROUND, linewidth=width, solid_capstyle='round')
