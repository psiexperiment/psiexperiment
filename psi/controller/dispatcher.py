'''
Single-owner dispatcher for the experiment control plane.

All experiment-control decisions (action matching/invocation, the action
context, delayed events) execute on one dedicated thread. Data-plane threads
(hardware acquisition callbacks, input pipelines) and the GUI thread request
control work through this dispatcher instead of mutating shared state
directly. See docs/threading.md for the full contract.

Design notes
------------
- ``submit_sync`` preserves the historical synchronous semantics of
  ``invoke_actions``: the caller blocks until the work completes, receives
  its return value, and sees its exceptions. When called *from* the
  dispatcher thread (an action triggering another action), the work runs
  inline so recursion cannot deadlock.
- ``submit`` is fire-and-forget for callers that must never block (e.g.,
  high-rate acquisition callbacks). Exceptions are logged.
- ``call_later`` schedules a named callable. The timer thread only enqueues:
  user code always runs on the dispatcher thread.
'''
import logging
log = logging.getLogger(__name__)

import queue
import threading
from concurrent.futures import Future


class ControlDispatcher:

    def __init__(self, name='control-dispatcher'):
        self._name = name
        self._queue = queue.SimpleQueue()
        self._thread = None
        self._thread_lock = threading.Lock()
        self._timers = {}
        self._timers_lock = threading.Lock()
        self._stopped = False

    ############################################################################
    # Worker thread management
    ############################################################################
    def _ensure_thread(self):
        # Started lazily so that constructing a plugin (e.g., in tests) does
        # not spawn a thread until control work is actually requested.
        if self._thread is None or not self._thread.is_alive():
            with self._thread_lock:
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(
                        target=self._run, name=self._name, daemon=True)
                    self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            fn, args, kwargs, future = item
            if future is None:
                try:
                    fn(*args, **kwargs)
                except Exception:
                    log.exception('Error in dispatched control-plane call %r', fn)
            else:
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as e:
                    future.set_exception(e)

    def is_dispatcher_thread(self):
        return threading.current_thread() is self._thread

    ############################################################################
    # Submitting work
    ############################################################################
    def submit_sync(self, fn, *args, **kwargs):
        '''
        Run `fn` on the dispatcher thread and block until it completes.

        Returns `fn`'s return value; exceptions propagate to the caller. If
        already on the dispatcher thread, runs inline (allowing actions to
        recursively invoke actions).
        '''
        if self._stopped:
            raise RuntimeError('ControlDispatcher has been stopped')
        self._ensure_thread()
        if self.is_dispatcher_thread():
            return fn(*args, **kwargs)
        future = Future()
        self._queue.put((fn, args, kwargs, future))
        return future.result()

    def submit(self, fn, *args, **kwargs):
        '''
        Run `fn` on the dispatcher thread without waiting (fire-and-forget).
        Exceptions are logged, not raised.
        '''
        if self._stopped:
            log.debug('Dispatcher stopped; dropping %r', fn)
            return
        self._ensure_thread()
        self._queue.put((fn, args, kwargs, None))

    ############################################################################
    # Named delayed calls (replaces ad-hoc threading.Timer usage)
    ############################################################################
    def call_later(self, name, delay, fn, *args, cancel_existing=True):
        '''
        Schedule `fn` to run on the dispatcher thread after `delay` seconds.

        A subsequent call with the same name cancels the pending one (unless
        `cancel_existing` is False).
        '''
        if cancel_existing:
            self.cancel(name)
        timer = threading.Timer(delay, self._fire, args=(name, fn, args))
        timer.daemon = True
        with self._timers_lock:
            self._timers[name] = timer
        timer.start()

    def _fire(self, name, fn, args):
        # Runs on the timer thread: discard the registration, then hand the
        # actual work to the dispatcher thread.
        with self._timers_lock:
            self._timers.pop(name, None)
        self.submit(fn, *args)

    def cancel(self, name):
        '''
        Cancel the named delayed call. No-op if it does not exist or has
        already fired.
        '''
        with self._timers_lock:
            timer = self._timers.pop(name, None)
        if timer is not None:
            timer.cancel()
            log.debug('Cancelled delayed call %s', name)

    def cancel_all(self):
        with self._timers_lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.cancel()

    ############################################################################
    # Shutdown
    ############################################################################
    def stop(self):
        self._stopped = True
        self.cancel_all()
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
