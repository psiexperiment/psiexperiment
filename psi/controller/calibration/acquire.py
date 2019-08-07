import time


class Acquire(object):

    def __init__(self, engine, queue, epoch_size):
        # Setup output
        self.ao_queue = queue
        self.ai_queue = queue.create_connection()
        self.ai_epochs = []
        self.engine = engine
        self.complete = False

        self.engine.register_ao_callback(self.ao_callback)

        # Setup input
        fs = engine.hw_ai_channels[0].fs
        ai_cb = extract_epochs(fs, self.ai_queue, epoch_size, epoch_size*100,
                               self.ai_callback)
        self.engine.register_ai_callback(ai_cb.send)

    def ao_callback(self, event):
        samples = event.engine.get_space_available(event.channel_name)
        waveform, empty = self.ao_queue.pop_buffer(samples)
        event.engine.append_hw_ao(waveform)

    def ai_callback(self, event):
        self.ai_epochs.append(event)
        if self.ao_queue.count_trials() == 0:
            if len(self.ai_queue) == 0:
                self.complete = True

    def start(self):
        self.engine.start()

    def join(self):
        while not self.complete:
            time.sleep(0.1)


def acquire(engine, queue, epoch_size):
    acq = Acquire(engine, queue, epoch_size)
    acq.start()
    acq.join()
    return acq.ai_epochs
