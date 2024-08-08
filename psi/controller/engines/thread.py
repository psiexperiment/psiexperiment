import logging
log = logging.getLogger(__name__)

import sys
from threading import Thread

from enaml.application import deferred_call

from psi import get_config


class DAQThread(Thread):

    def __init__(self, poll_interval, stop_requested, callback, name):
        log.debug('Initializing acquisition thread')
        super().__init__()
        self.poll_interval = poll_interval
        self.stop_requested = stop_requested
        self.callback = callback
        self.name = name

    def run(self):
        # This is a rather complicated piece of code because we need to
        # override the threading module's built-in exception handling as well
        # as defe the exception back to the main thread (where it will properly
        # handle exceptions). If we call psi.application.exception_handler
        # directly from the thread, it will not have access to the application
        # instance (or workspace).
        try:
            self._run()
        except:
            log.info('Caught exception')
            deferred_call(sys.excepthook, *sys.exc_info())

    def _run(self):
        profile = get_config('PROFILE', False)
        if profile:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()

        log.debug('Starting acquisition thread')
        while not self.stop_requested.wait(self.poll_interval):
            stop = self.callback()
            if stop:
                break

        if profile:
            pr.disable()
            path = get_config('LOG_ROOT') / f'{self.name}_thread.pstat'
            pr.dump_stats(path)

        log.debug('Exiting acquistion thread')
