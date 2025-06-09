import time
import matplotlib.pyplot as plt
import numpy as np

from .playrec import PlayRec


def test_delay(fs=96e3, output_channel=1, input_channel=13):
    # 1 to 4 for hardwired
    # This can be used to characterize the delays in the system
    data = []
    signal = []

    def ai_cb(s0, samples, d):
        nonlocal data
        data.append(d)

    def ao_cb(s0, samples, buffer):
        nonlocal signal
        t = (np.arange(samples) + s0) / fs
        w = 0.001 * np.sin(2 * np.pi * 1e3 * t)
        w = np.zeros((1, samples))
        if len(signal) == 0:
            w[:, :100] = 1
        signal.append(w)
        buffer[:] = w
        #return w[np.newaxis]

    dev = PlayRec(fs, 'ASIO Fireface USB', [input_channel],
                  [output_channel], ai_cb, ao_cb, blocksize=0)

    dev.start()
    time.sleep(0.5)
    dev.stop()


    data = np.concatenate(data, axis=-1)
    signal = np.concatenate(signal, axis=-1)

    print(dev.stream.latency)

    try:
        input_delay = np.flatnonzero(data[0] > 0.001)[0]
        print(f'Actual AO to AI delay: {input_delay / fs * 1e3:2f} ms')
        print(f'Actual delay in samples: {input_delay}')
    except:
        print('Error determining delay. Maybe not connected correctly?')

    plt.plot(signal[0], label='signal')
    plt.plot(data[0], label='input')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_delay()
