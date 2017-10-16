import time

from .input import extract_epochs


class Acquire(object):

    def __init__(self, engine, queue, epoch_size):
        # Setup output
        self.n_epochs = queue.count_trials()
        self.ai_queue = queue.create_connection()
        self.ai_epochs = []
        self.engine = engine
        self.complete = False

        # Setup input
        ai_fs = engine.hw_ai_channels[0].fs
        ai_cb = extract_epochs(ai_fs, self.ai_queue, epoch_size, epoch_size*10,
                               0, '', self.ai_callback)
        self.engine.register_ai_callback(ai_cb.send)

    def ai_callback(self, event):
        self.ai_epochs.extend(event)
        if len(self.ai_epochs) == self.n_epochs:
            self.complete = True

    def start(self):
        self.engine.start()

    def join(self):
        while not self.complete:
            time.sleep(0.1)
        self.engine.stop()


def acquire(engine, queue, epoch_size):
    acq = Acquire(engine, queue, epoch_size)
    acq.start()
    acq.join()
    return acq.ai_epochs
