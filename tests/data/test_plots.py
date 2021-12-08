from psi.data.plots import make_color


def test_make_color():
    make_color('red')
    make_color('seagreen')
    make_color((0, 0, 0, 0))
    make_color((0, 0, 0))
