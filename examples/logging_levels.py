import logging


if __name__ == '__main__':
    from psi.application import configure_logging
    configure_logging('TRACE')

    # Logger has to be created *after* logging is configured for this to work.
    log = logging.getLogger(__name__)

    for level in ('trace', 'debug', 'info', 'warn', 'error', 'critical'):
        fn = getattr(log, level)
        fn(f'This is a {level} message')
