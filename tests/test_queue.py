import pytest

from collections import Counter, deque

import enaml
import numpy as np

with enaml.imports():
    from psi.controller.calibration.api import FlatCalibration
    from psi.controller.api import (extract_epochs, FIFOSignalQueue,
                                    InterleavedFIFOSignalQueue)
    from psi.token.primitives import Cos2EnvelopeFactory, ToneFactory

#fs = 100e3
fs = 195312.5
isi = np.round(1/76.0, 5)


def make_tone(frequency=250, duration=5e-3):
    calibration = FlatCalibration.as_attenuation()
    tone = ToneFactory(fs, 0, frequency, 0, 1, calibration)
    return Cos2EnvelopeFactory(fs, 0, 0.5e-3, duration, tone)


def make_queue(ordering, frequencies, trials, duration=5e-3, isi=isi):
    if ordering == 'FIFO':
        queue = FIFOSignalQueue(fs)
    elif ordering == 'interleaved':
        queue = InterleavedFIFOSignalQueue(fs)
    else:
        raise ValueError(f'Unrecognized queue ordering {ordering}')

    conn = deque()
    queue.connect(conn.append, 'added')
    removed_conn = deque()
    queue.connect(removed_conn.append, 'removed')

    keys = []
    tones = []
    for frequency in frequencies:
        t = make_tone(frequency=frequency, duration=duration)
        delay = max(isi-duration, 0)
        k = queue.append(t, trials, delay)
        keys.append(k)
        tones.append(t)

    return queue, conn, removed_conn, keys, tones


def test_long_tone_queue():
    queue, conn, rem_conn, (k1, k2), (t1, t2) = \
        make_queue('interleaved', [1e3, 5e3], 5, duration=1, isi=1)

    waveforms = []
    n_pop = round(fs * 0.25)
    for i in range(16):
        w = queue.pop_buffer(n_pop)
        waveforms.append(w)

    waveforms = np.concatenate(waveforms, axis=-1)
    waveforms.shape = 4, -1
    assert waveforms.shape == (4, round(fs))
    assert np.all(waveforms[0] == waveforms[2])
    assert np.all(waveforms[1] == waveforms[3])
    assert np.any(waveforms[0] != waveforms[1])


def test_fifo_queue_pause_with_requeue():
    # Helper function to track number of remaining keys
    def _adjust_remaining(k1, k2, n):
        nk1 = min(k1, n)
        nk2 = min(n-nk1, k2)
        return k1-nk1, k2-nk2

    queue, conn, rem_conn, (k1, k2), (t1, t2) = \
        make_queue('FIFO', [1e3, 5e3], 100)
    extractor_conn = deque()
    extractor_rem_conn = deque()
    queue.connect(extractor_conn.append, 'added')
    queue.connect(extractor_rem_conn.append, 'removed')

    # Generate the waveform template
    n_t1 = t1.get_remaining_samples()
    n_t2 = t2.get_remaining_samples()
    t1_waveform = t1.next(n_t1)
    t2_waveform = t2.next(n_t2)

    waveforms = []
    extractor = extract_epochs(fs=fs,
                               queue=extractor_conn,
                               removed_queue=extractor_rem_conn,
                               poststim_time=0,
                               buffer_size=0,
                               epoch_size=15e-3,
                               target=waveforms.extend)

    # Track number of trials remaining
    k1_left, k2_left = 100, 100
    samples = int(round(fs))

    # Since the queue uses the delay (between offset and onset of
    # consecutive segments), we need to calculate the actual ISI since it
    # may have been rounded to the nearest sample.
    delay_samples = round((isi-t1.duration) * fs)
    duration_samples = round(t1.duration * fs)
    total_samples = duration_samples + delay_samples
    actual_isi = total_samples / fs

    ###########################################################################
    # First, queue up 2 seconds worth of trials
    ###########################################################################
    waveform = queue.pop_buffer(samples * 2)
    n_queued = np.floor(2 / actual_isi) + 1
    t1_lb = 0
    t2_lb = 100 * total_samples
    t2_lb = int(t2_lb)
    assert np.all(waveform[t1_lb:t1_lb+duration_samples] == t1_waveform)
    assert np.all(waveform[t2_lb:t2_lb+duration_samples] == t2_waveform)

    assert len(conn) == np.ceil(2 / actual_isi)
    assert len(rem_conn) == 0
    keys = [i['key'] for i in conn]
    assert set(keys) == {k1, k2}
    assert set(keys[:100]) == {k1}
    assert set(keys[100:]) == {k2}

    k1_left, k2_left = _adjust_remaining(k1_left, k2_left, n_queued)
    assert queue.remaining_trials(k1) == k1_left
    assert queue.remaining_trials(k2) == k2_left
    conn.clear()

    ###########################################################################
    # Now, pause
    ###########################################################################
    # Pausing should remove all epochs queued up after 0.5s. After sending
    # the first waveform to the extractor, we generate a new waveform to
    # verify that no additional trials are queued and send that to the
    # extractor.
    queue.pause(round(0.5 * fs) / fs)
    extractor.send(waveform[:round(0.5*fs)])

    # We need to add 1 to account for the very first trial.
    n_queued = int(np.floor(2 / actual_isi)) + 1
    n_kept = int(np.floor(0.5 / actual_isi)) + 1

    # Now, fix the counters
    k1_left, k2_left = _adjust_remaining(100, 100, n_kept)

    # This is the total number that were removed when we paused.
    n_removed = n_queued - n_kept

    # Subtract 1 because we haven't fully captured the last trial that
    # remains in the queue because the epoch_size was chosen such that the
    # end of the epoch to be extracted is after 0.5s.
    n_captured = n_kept - 1
    assert len(waveforms) == n_captured

    # Doing this will capture the final epoch.
    waveform = queue.pop_buffer(samples)
    assert np.all(waveform == 0)
    extractor.send(waveform)
    assert len(waveforms) == (n_captured + 1)

    # Verify removal event is properly notifying the timestamp
    rem_t0 = np.array([i['t0'] for i in rem_conn])
    assert np.all(rem_t0 >= 0.5)
    assert (rem_t0[0] % actual_isi) == pytest.approx(0, 0.1/fs)

    assert queue.remaining_trials(k1) == k1_left
    assert queue.remaining_trials(k2) == k2_left
    assert len(conn) == 0
    assert len(rem_conn) == n_removed

    rem_count = Counter(i['key'] for i in rem_conn)
    assert rem_count[k1] == 100-n_kept
    assert rem_count[k2] == n_queued-100
    conn.clear()
    rem_conn.clear()

    queue.resume(samples * 1.5 / fs)

    waveform = queue.pop_buffer(samples)
    n_queued = np.floor(1 / actual_isi) + 1
    k1_left, k2_left = _adjust_remaining(k1_left, k2_left, n_queued)

    extractor.send(waveform)
    print('N', len(waveforms))
    print('EQ2??', np.all(waveforms[38]['signal'][:n_t1] == t1_waveform))

    assert len(conn) == np.floor(1/actual_isi) + 1
    assert queue.remaining_trials(k1) == k1_left
    assert queue.remaining_trials(k2) == k2_left
    assert len(conn) == np.floor(1/actual_isi) + 1
    keys += [i['key'] for i in conn]
    conn.clear()

    waveform = queue.pop_buffer(5*samples)
    n_queued = np.floor(5/actual_isi) + 1
    k1_left, k2_left = _adjust_remaining(k1_left, k2_left, n_queued)

    extractor.send(waveform)
    assert queue.remaining_trials(k1) == k1_left
    assert queue.remaining_trials(k2) == k2_left
    keys += [i['key'] for i in conn]

    # We requeued 1.5 second worth of trials so need to factor this because
    # keys (from conn) did not remove the removed keys.
    assert len(keys) == (200 + n_removed)

    # However, the extractor is smart enough to handle cancel appropriately
    # and should only have the 200 we originally intended.
    assert len(waveforms) == 200

    # This should capture the 1-sample bug that sometimes occurs when using
    # int() instead of round() with quirky sample rates (e.g., like with the
    # RZ6).
    n = len(t1_waveform)
    t1_waveforms = np.vstack(w['signal'] for w in waveforms[:100])[..., :n]
    t2_waveforms = np.vstack(w['signal'] for w in waveforms[100:])[..., :n]
    # discrepancy happens at 38
    #import matplotlib.pyplot as plt
    #plt.plot(t1_waveforms[30], 'k')
    #plt.plot(t1_waveform, 'r')
    #plt.show()

    assert np.all(t1_waveforms == t1_waveform)
    assert np.all(t2_waveforms == t2_waveform)

    #for waveform in waveforms[:100]:
    #    assert np.all(waveform['signal'][..., :n] == t1_waveform)
    #for waveform in waveforms[100:]:
    #    assert np.all(waveform['signal'][..., :n] == t2_waveform)


def test_queue_isi_with_pause():
    '''
    Verifies that queue generates samples at the expected ISI and also verifies
    pause functionality works as expected.
    '''
    queue, conn, _, _, (t1,) = make_queue('FIFO', [250], 500)
    duration = 1
    samples = round(duration*fs)
    queue.pop_buffer(samples)
    expected_n = int(duration / isi) + 1
    assert len(conn) == expected_n

    queue.pause()
    waveform = queue.pop_buffer(samples)
    assert np.sum(waveform**2) == 0
    assert len(conn) == int(duration / isi) + 1
    queue.resume()

    queue.pop_buffer(samples)
    assert len(conn) == np.ceil(2 * duration / isi)
    queue.pop_buffer(samples)
    assert len(conn) == np.ceil(3 * duration / isi)

    times = [u['t0'] for u in conn]
    assert times[0] == 0
    all_isi = np.diff(times)

    # Since the queue uses the delay (between offset and onset of
    # consecutive segments), we need to calculate the actual ISI since it
    # may have been rounded to the nearest sample.
    actual_isi = round((isi-t1.duration)*fs) / fs + t1.duration

    # We paused the playout, so this means that we have a very long delay in
    # the middle of the queue. Check for this delay, ensure that there's only
    # one ISI with this delay and then verify that all other ISIs are the
    # expected ISI given the tone pip duration.
    expected_max_isi = round((duration + actual_isi) * fs) / fs
    assert all_isi.max() == expected_max_isi
    m = all_isi == all_isi.max()
    assert sum(m) == 1

    # Now, check that all other ISIs are as expected.
    expected_isi = round(actual_isi * fs) / fs
    np.testing.assert_almost_equal(all_isi[~m], expected_isi)


def test_fifo_queue_pause_resume_timing():
    trials = 20
    samples = int(fs)
    queue, conn, _, _, _ = make_queue('FIFO', (1e3, 5e3), trials)
    queue.pop_buffer(samples)
    conn.clear()
    queue.pause(0.1025)
    queue.pop_buffer(samples)
    queue.resume(0.6725)
    queue.pop_buffer(samples)
    t0 = [i['t0'] for i in conn]
    assert t0[0] == round(0.6725 * fs) / fs


def test_fifo_queue_ordering():
    trials = 20
    samples = round(fs)

    queue, conn, _, (k1, k2), (t1, _) = \
        make_queue('FIFO', (1e3, 5e3), trials)
    epoch_samples = round(t1.duration * fs)

    waveforms = []
    queue_empty = False
    def mark_empty():
        nonlocal queue_empty
        queue_empty = True

    extractor = extract_epochs(fs=fs,
                               queue=conn,
                               epoch_size=None,
                               poststim_time=0,
                               buffer_size=0,
                               target=waveforms.extend,
                               empty_queue_cb=mark_empty)

    waveform = queue.pop_buffer(samples)
    extractor.send(waveform)
    assert queue_empty

    metadata = list(conn)
    for md in metadata[:trials]:
        assert k1 == md['key']
    for md in metadata[trials:]:
        assert k2 == md['key']

    waveforms = np.vstack([w['signal'] for w in waveforms])
    assert waveforms.shape == (trials * 2, epoch_samples)
    for w in waveforms[:trials]:
        assert np.all(w == waveforms[0])
    for w in waveforms[trials:]:
        assert np.all(w == waveforms[trials])
    assert np.any(waveforms[0] != waveforms[trials])


def test_interleaved_fifo_queue_ordering():
    samples = round(fs)
    trials = 20

    queue, conn, _, (k1, k2), (t1, _) = \
        make_queue('interleaved', (1e3, 5e3), trials)
    epoch_samples = round(t1.duration * fs)

    waveforms = []
    queue_empty = False
    def mark_empty():
        nonlocal queue_empty
        queue_empty = True

    extractor = extract_epochs(fs=fs,
                               queue=conn,
                               epoch_size=None,
                               poststim_time=0,
                               buffer_size=0,
                               target=waveforms.extend,
                               empty_queue_cb=mark_empty)

    waveform = queue.pop_buffer(samples)
    extractor.send(waveform)
    assert queue_empty

    # Verify that keys are ordered properly
    metadata = list(conn)
    for md in metadata[::2]:
        assert k1 == md['key']
    for md in metadata[1::2]:
        assert k2 == md['key']

    waveforms = np.vstack([w['signal'] for w in waveforms])
    assert waveforms.shape == (trials * 2, epoch_samples)
    for w in waveforms[::2]:
        assert np.all(w == waveforms[0])
    for w in waveforms[1::2]:
        assert np.all(w == waveforms[1])
    assert np.any(waveforms[0] != waveforms[1])


def test_queue_continuous_tone():
    '''
    Test ability to work with continuous tones and move to the next one
    manually (e.g., as in the case of DPOAEs).
    '''
    samples = round(1*fs)
    queue, conn, _, _, (t1, t2) = make_queue('FIFO', (1e3, 5e3), 1, duration=100)

    # Get samples from t1
    assert queue.get_max_duration() == 100
    assert np.all(queue.pop_buffer(samples) == t1.next(samples))
    assert np.all(queue.pop_buffer(samples) == t1.next(samples))

    # Switch to t2
    queue.next_trial()
    assert np.all(queue.pop_buffer(samples) == t2.next(samples))
    assert np.all(queue.pop_buffer(samples) == t2.next(samples))

    # Ensure timing information correct
    assert len(conn) == 2
    assert conn.popleft()['t0'] == 0
    assert conn.popleft()['t0'] == (samples * 2) / fs


def test_future_pause():
    queue, conn, rem_conn, (k1, k2), (t1, t2) = \
        make_queue('FIFO', [1e3, 5e3], 100)
    queue.pop_buffer(1000)
    # This is OK
    queue.pause(1000 / fs)
    queue.resume(1000 / fs)
    # This is not
    with pytest.raises(ValueError):
        queue.pause(1001 / fs)


def test_queue_partial_capture():
    queue, conn, rem_conn, (k1, k2), (t1, t2) = \
        make_queue('FIFO', [1e3, 5e3], 100)
    extractor_conn = deque()
    extractor_rem_conn = deque()
    queue.connect(extractor_conn.append, 'added')
    queue.connect(extractor_rem_conn.append, 'removed')

    waveforms = []
    extractor = extract_epochs(fs=fs,
                               queue=extractor_conn,
                               removed_queue=extractor_rem_conn,
                               poststim_time=0,
                               buffer_size=0,
                               epoch_size=15e-3,
                               target=waveforms.extend)

    samples = int(fs)
    tone_samples = t1.get_remaining_samples()
    w1 = queue.pop_buffer(tone_samples / 2)
    queue.pause(0.5 * tone_samples / fs)
    w2 = queue.pop_buffer(samples)
    extractor.send(w1)
    extractor.send(w2)

    assert len(waveforms) == 0
