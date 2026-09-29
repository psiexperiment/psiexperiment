'''
Tests for how the websocket client reports a failed connection.

The client runs in a daemon thread, and `start_thread` used to return the
moment it was started, without checking the outcome. A connection that
could not be made therefore raised inside that thread, printed a
traceback, and left the experiment running with no channel back to the
launcher that started it -- reporting nothing and answering nothing.

That is not hypothetical. The launcher wrote the address under the
variable's old name for several commits after it was renamed, so every
cfts experiment ran disconnected, and nothing said so.
'''
import enaml
import pytest

with enaml.imports():
    from psi.paradigms.core.websocket_mixins import WebsocketClientPlugin


@pytest.fixture
def plugin():
    return WebsocketClientPlugin()


def test_no_address_does_not_connect(plugin):
    '''
    Running a paradigm by hand rather than through a launcher is a normal
    thing to do, and must not fail or start a thread.
    '''
    plugin.uri = ''
    plugin.start_thread()
    assert plugin.thread is None
    assert plugin.error is None


def test_no_address_survives_send_and_disconnect(plugin):
    '''
    The experiment still emits events and still shuts down. Neither may
    trip over the connection that was never made -- disconnect used to
    join a thread that does not exist.
    '''
    plugin.uri = ''
    plugin.start_thread()
    plugin.send_message({'event': 'experiment_start'})
    plugin._disconnect()
    assert plugin.send_queue.qsize() == 0


def test_unreachable_address_raises_on_the_calling_thread(plugin):
    '''
    An address *was* supplied, so something meant this experiment to have
    a channel back. Failing to get one is reported here, where it stops
    the experiment, rather than in the daemon thread where nobody sees it.
    '''
    # Port 1 on localhost: nothing listens there, and it fails fast.
    plugin.uri = 'ws://127.0.0.1:1'
    with pytest.raises(RuntimeError, match='Could not reach the websocket'):
        plugin.start_thread()
    assert plugin.error is not None


def test_malformed_address_raises_on_the_calling_thread(plugin):
    '''
    The failure mode the rename actually produced was a URI that is not a
    URI at all.
    '''
    plugin.uri = 'not-a-uri'
    with pytest.raises(RuntimeError, match='Could not reach the websocket'):
        plugin.start_thread()


def test_failed_connection_leaves_nothing_to_join(plugin):
    '''
    A failed attempt must not leave a dead thread behind for disconnect
    to join at the end of the experiment.
    '''
    plugin.uri = 'ws://127.0.0.1:1'
    with pytest.raises(RuntimeError):
        plugin.start_thread()
    plugin._disconnect()
