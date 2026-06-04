class SuspensionError(Exception):
    """Raised by a Middleware to suspend the agentic loop.

    The exception propagates to the caller, who is responsible for handling the
    suspension — typically by persisting state and resuming the loop later.

    Args:
        details: Optional context about why the loop was suspended (e.g. a
            checkpoint object or status message). Available as ``self.details``.
        *args: Forwarded to ``Exception.__init__``.

    Example::

        raise SuspensionError({"checkpoint": state}, "human approval required")
    """

    def __init__(self, details: object = None, *args):
        self.details = details
        super().__init__(*args)

    def __str__(self):
        if self.details is None:
            return super().__str__()
        return f"LoopSuspensionError: The Agent Loop was Suspended with the following details: {self.details}"
