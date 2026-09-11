"""Cancellable Qt background jobs."""

from __future__ import annotations

import traceback
from threading import Event

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    progress = Signal(float, str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class FunctionWorker(QRunnable):
    def __init__(self, function, *args, with_callbacks: bool = False, **kwargs) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.with_callbacks = with_callbacks
        self.signals = WorkerSignals()
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            kwargs = dict(self.kwargs)
            if self.with_callbacks:
                kwargs["progress"] = self.signals.progress.emit
                kwargs["cancelled"] = self.cancel_event.is_set
            value = self.function(*self.args, **kwargs)
            self.signals.result.emit(value)
        except InterruptedError:
            self.signals.error.emit("Analysis cancelled")
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        finally:
            self.signals.finished.emit()

