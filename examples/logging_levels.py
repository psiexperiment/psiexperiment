'''
This illustrates the use of an additional `trace` level that has been
incorporated into psiexperiment. The `trace` level is typically used for
tracing input-output calls to the hardware. Since hundreds of logging records
may be emitted per second when set to the `trace` level, logging should
typically be set to `debug` or higher.
'''

import logging


if __name__ == '__main__':
    from psi.application import configure_logging
    configure_logging('TRACE')

    # Logger has to be created *after* logging is configured for this to work.
    log = logging.getLogger(__name__)

    for level in ('trace', 'debug', 'info', 'warning', 'error', 'critical'):
        fn = getattr(log, level)
        fn(f'This is a {level} message')
