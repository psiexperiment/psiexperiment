from psi.application import is_benign_qt_message


def test_window_geometry_clamp_is_benign():
    # Emitted by Windows when a window is taller than the monitor (seen with
    # the ABTS go-nogo STM paradigm on a 1080p screen).
    message = (
        'QWindowsWindow::setGeometry: Unable to set geometry 959x1246+961+31 '
        '(frame: 975x1285+953+0) on QWidgetWindow/"obj-2964613128320Window" '
        'on "HL269DPB". Resulting geometry: 959x1061+961+31'
    )
    assert is_benign_qt_message(message)


def test_drag_leave_without_enter_is_benign():
    # Seen over a plot while a cfts click ABR was running; it stopped the
    # experiment.
    message = ('QGraphicsView::dragLeaveEvent: drag leave received before '
               'drag enter')
    assert is_benign_qt_message(message)


def test_cross_thread_warning_is_not_benign():
    message = ('QObject::setParent: Cannot set parent, new parent is in a '
               'different thread')
    assert not is_benign_qt_message(message)
