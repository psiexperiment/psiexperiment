'''
Tests for `psi.controller.calibration.acquire.wait_for_epochs`: acquisition
must not stop until every input has all of its epochs.
'''
import threading
import time

from psi.controller.calibration.acquire import wait_for_epochs


class FakeQueue:

    def __init__(self, empty=True):
        self.empty = empty

    def is_empty(self):
        return self.empty


class FakeEpochInput:

    def __init__(self, complete=False):
        self.complete = complete


def wait_in_thread(queue, inputs):
    thread = threading.Thread(target=wait_for_epochs,
                              args=(queue, inputs, 0.001), daemon=True)
    thread.start()
    return thread


def test_returns_when_everything_is_complete():
    wait_for_epochs(FakeQueue(), [FakeEpochInput(True), FakeEpochInput(True)])


def test_waits_for_every_input():
    # The last input finishing first used to end the acquisition while the
    # others still had epochs to extract.
    first, last = FakeEpochInput(), FakeEpochInput(complete=True)
    thread = wait_in_thread(FakeQueue(), [first, last])
    time.sleep(0.05)
    assert thread.is_alive()
    first.complete = True
    thread.join(1)
    assert not thread.is_alive()


def test_waits_for_the_queue():
    queue = FakeQueue(empty=False)
    thread = wait_in_thread(queue, [FakeEpochInput(True)])
    time.sleep(0.05)
    assert thread.is_alive()
    queue.empty = True
    thread.join(1)
    assert not thread.is_alive()
