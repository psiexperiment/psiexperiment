'''
Exception hierarchy for psiexperiment.

Catching :class:`PSIException` distinguishes errors raised (and understood)
by the framework from arbitrary programming errors.
'''


class PSIException(Exception):
    '''
    Base class for all exceptions raised by psiexperiment itself.
    '''


class ActionError(PSIException):
    '''
    Raised when an ExperimentAction's command or callback fails.

    The original exception is available as ``__cause__``.
    '''

    def __init__(self, message, action=None, event=None):
        super().__init__(message)
        #: The ExperimentActionBase instance that failed.
        self.action = action
        #: Name of the event that triggered the action.
        self.event = event
