import pytest

pytest.importorskip('matplotlib')
Image = pytest.importorskip('PIL.Image')

from matplotlib.colors import to_rgb

from psi.launcher import icons


def rgb(color):
    return tuple(round(c * 255) for c in to_rgb(color))


def pixel(path, xy):
    return Image.open(path).convert('RGB').getpixel(xy)


def close_to(actual, expected, tolerance=2):
    return all(abs(a - e) <= tolerance for a, e in zip(actual, expected))


def test_png_has_frame_around_background(tmp_path):
    path = icons.make_icon(lambda ax: None, tmp_path / 'icon.png')
    assert Image.open(path).size == (icons.PNG_SIZE, icons.PNG_SIZE)
    assert close_to(pixel(path, (0, 0)), rgb(icons.FOREGROUND))
    assert close_to(pixel(path, (128, 128)), rgb(icons.BACKGROUND))


def test_ico_is_optional(tmp_path):
    icons.make_icon(lambda ax: None, tmp_path / 'icon.png')
    assert [p.name for p in tmp_path.iterdir()] == ['icon.png']


def test_ico_holds_every_size(tmp_path):
    icons.make_icon(lambda ax: None, tmp_path / 'icon.png', tmp_path / 'icon.ico')
    sizes = Image.open(tmp_path / 'icon.ico').info['sizes']
    assert sorted(sizes) == sorted(icons.ICO_SIZES)


def test_draw_sees_final_limits(tmp_path):
    seen = {}

    def draw(ax):
        seen['xlim'] = ax.get_xlim()
        seen['ylim'] = ax.get_ylim()

    icons.make_icon(draw, tmp_path / 'icon.png', xlim=(-12, 22), ylim=(-22, 12))
    assert seen == {'xlim': (-12, 22), 'ylim': (-22, 12)}


def test_drawing_does_not_rescale(tmp_path):
    # Data far outside the limits must not pull the axes out to fit it, or
    # the motif would shrink.
    limits = {}

    def draw(ax):
        ax.plot([-100, 100], [-100, 100])
        limits['ylim'] = ax.get_ylim()

    icons.make_icon(draw, tmp_path / 'icon.png')
    assert limits['ylim'] == (-1.5, 1.5)


def test_plot_signal_fills_below_trace(tmp_path):
    def draw(ax):
        icons.plot_signal(ax, [-0.05, 1.05], [0, 0])

    path = icons.make_icon(draw, tmp_path / 'icon.png')
    # y = 0 is the middle of the default ylim, so the trace runs through the
    # middle row with the fill below it and the background above.
    assert close_to(pixel(path, (128, 128)), rgb(icons.FOREGROUND))
    assert close_to(pixel(path, (128, 200)), rgb(icons.FILL))
    assert close_to(pixel(path, (128, 56)), rgb(icons.BACKGROUND))


def test_frame_covers_motif(tmp_path):
    # A motif reaching the edge of the icon disappears under the frame.
    def draw(ax):
        ax.axhspan(-1.5, 1.5, color=icons.FILL)

    path = icons.make_icon(draw, tmp_path / 'icon.png')
    assert close_to(pixel(path, (0, 128)), rgb(icons.FOREGROUND))
    assert close_to(pixel(path, (128, 128)), rgb(icons.FILL))
