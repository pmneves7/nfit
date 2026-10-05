"""Cooperative progress and cancellation for scientific I/O operations."""

from contextlib import contextmanager
from contextvars import ContextVar

_REPORTER = ContextVar("nfit_operation_reporter", default=None)


@contextmanager
def operation_progress(callback):
    """Attach a narrow progress callback to this operation, including archive I/O."""
    token = _REPORTER.set(callback)
    try:
        yield
    finally:
        _REPORTER.reset(token)


def report_operation(message, *, completed=0, total=0):
    """Report bounded work; the callback may raise to cancel before publication."""
    callback = _REPORTER.get()
    if callback is not None:
        callback(dict(stage="archive_io", message=message, iteration=completed, total=total))
