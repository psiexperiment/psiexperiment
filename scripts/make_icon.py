# Generates psi/experiment/psi-logo.png, the window icon loaded by
# psi.experiment.util.load_icon. Run from anywhere:
#   python scripts/make_icon.py
#
# The frame, palette and output sizes come from psiapp.icons, shared with the
# launchers built on psi, so that psi reads as part of the same family. That
# makes psiapp (pip install psiapp[icons]) a requirement for running this
# script only -- psi itself does not, and must not, depend on psiapp, which is
# built on psi.
#
# Only the motif is drawn here: psi's circuit-board trident, as white traces
# ending in cornflowerblue pads. The white frame stands in for the circuit box
# that surrounded the trident in the original logo.
from pathlib import Path

from matplotlib.patches import Circle

from psiapp.icons import FILL, FOREGROUND, TRACE_WIDTH, make_icon


OUTPUT = Path(__file__).parents[1] / 'psi' / 'experiment' / 'psi-logo.png'

#: Radius of the pads at the end of each trace, in data units (same as the
#: original logo).
PAD_RADIUS = 2


def draw(ax):
    # The trident: a stem running from the bottom pad to the top pad, and a
    # fork whose arms step outward to the two side pads. Square caps and
    # mitered joins keep the traces looking like etched copper rather than
    # pen strokes.
    trace = dict(color=FOREGROUND, lw=TRACE_WIDTH, solid_capstyle='butt',
                 solid_joinstyle='miter', zorder=4)
    ax.plot([5, 5], [-15, 5], **trace)
    ax.plot([0, 0, 2.5, 7.5, 10, 10], [0, -5, -7.5, -7.5, -5, 0], **trace)
    for xy in [(0, 0), (5, 5), (10, 0), (5, -15)]:
        pad = Circle(xy, radius=PAD_RADIUS, lw=TRACE_WIDTH,
                     edgecolor=FOREGROUND, facecolor=FILL, zorder=5)
        ax.add_patch(pad)


if __name__ == '__main__':
    # Centered on the trident, which spans x = 0 to 10 and y = -15 to 5, with
    # enough room that the pads stay clear of the frame. Both ranges are the
    # same length so the pads come out round.
    make_icon(draw, OUTPUT, xlim=(-12, 22), ylim=(-22, 12))
