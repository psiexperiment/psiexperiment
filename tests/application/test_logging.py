import logging
from psi.application import configure_logging


def test_logging_trace_level_trace(caplog):
    # This verifies that we have properly configured the trace level and it
    # shows up when configuring logging to trace.
    configure_logging('TRACE')
    log = logging.getLogger(__name__)
    log.trace('This is a trace message')
    assert caplog.records[0].levelname == 'TRACE'


def test_logging_trace_level_info(caplog):
    # This verifies that we have properly configured the trace level and it
    # does not show up when logging is set to info.
    configure_logging('INFO')
    log = logging.getLogger(__name__)
    log.trace('This is a trace message')
    assert len(caplog.records) == 0
