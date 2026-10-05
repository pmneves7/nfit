"""Run one guarded scientific I/O job without blocking Qt painting or cancellation."""

from threading import Event

from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

from .operation_control import operation_progress
from .qt_operation_guard import GuiOperationGuard


class ResourceJobCancelled(Exception):
    """The user cancelled a resource operation before it was published."""


def run_resource_job(parent, title, task, *, on_progress=None):
    """Run ``task(progress_callback)`` on one worker and return its result.

    A nested Qt event loop keeps the modal progress window responsive. Project
    editing is guarded until the worker joins, including cancellation cleanup.
    Exceptions retain their original type for memory-budget handling.
    """
    cancel = Event()
    dialog = QtWidgets.QProgressDialog(title, "Cancel", 0, 0, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setToolTip("Cancel safely; the current archive is kept until a replacement is complete.")
    dialog.canceled.connect(cancel.set)
    outcome = []
    loop = QtCore.QEventLoop()

    class Worker(QtCore.QObject):
        progress = QtCore.Signal(dict)
        completed = QtCore.Signal(object)

        @QtCore.Slot()
        def run(self):
            def report(event):
                if cancel.is_set():
                    raise ResourceJobCancelled("Operation cancelled.")
                self.progress.emit(event)
            try:
                with operation_progress(report):
                    result = task(report)
                self.completed.emit((True, result))
            except BaseException as exc:
                self.completed.emit((False, exc))

    class Receiver(QtCore.QObject):
        @QtCore.Slot(dict)
        def progress(self, event):
            if not isValid(dialog):
                cancel.set()
                return
            dialog.setLabelText(str(event.get("message", title)))
            if on_progress is not None:
                try:
                    on_progress(event)
                except Exception:
                    cancel.set()

        @QtCore.Slot(object)
        def finish(self, result):
            outcome.append(result)
            thread.quit()

    thread = QtCore.QThread()
    worker = Worker()
    # Do not parent the completion receiver to a transient active window.
    # A deferred Qt deletion of that window must not strand the worker loop.
    receiver = Receiver()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.progress.connect(receiver.progress)
    worker.completed.connect(receiver.finish)
    worker.completed.connect(worker.deleteLater)
    thread.finished.connect(loop.quit)
    with GuiOperationGuard():
        dialog.show()
        thread.start()
        loop.exec()
        thread.wait()
        if isValid(dialog):
            dialog.close()
    thread.deleteLater()
    if not outcome:
        raise RuntimeError("Resource worker stopped without a result.")
    success, value = outcome[0]
    if not success:
        raise value
    return value
