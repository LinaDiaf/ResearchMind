"""Background execution of investigations for the Streamlit app.

Streamlit reruns the page script on every interaction, so the agent runs in a thread.
Progress is not kept in memory: every event is written to the database by the runner and the
page reads it from there, so progress survives page reloads and the run can be resumed after
a restart from its checkpoint.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable

from researchmind.log import get_logger

log = get_logger(__name__)


class BackgroundRuns:
    def __init__(self) -> None:
        self._threads: dict[uuid.UUID, threading.Thread] = {}
        self._errors: dict[uuid.UUID, str] = {}
        self._lock = threading.Lock()

    def start(self, investigation_id: uuid.UUID, fn: Callable[[], object]) -> bool:
        with self._lock:
            t = self._threads.get(investigation_id)
            if t and t.is_alive():
                return False

            def target() -> None:
                try:
                    fn()
                except Exception as exc:  # recorded on the investigation by the runner too
                    self._errors[investigation_id] = f"{type(exc).__name__}: {exc}"
                    log.warning("background_run_failed", investigation_id=str(investigation_id))

            self._errors.pop(investigation_id, None)
            t = threading.Thread(target=target, name=f"investigation-{investigation_id}", daemon=True)
            self._threads[investigation_id] = t
            t.start()
            return True

    def is_running(self, investigation_id: uuid.UUID) -> bool:
        t = self._threads.get(investigation_id)
        return bool(t and t.is_alive())

    def error(self, investigation_id: uuid.UUID) -> str | None:
        return self._errors.get(investigation_id)
