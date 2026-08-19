"""Execute a Plan, streaming output to whatever front end is watching.

Deliberately Qt-free: the CLI drives it directly and the GUI wraps it in a
QThread. One execution path means the two front ends cannot drift apart.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from .pipeline import Plan
from .stages import Step

LogFn = Callable[[str], None]
StepStartFn = Callable[[Step, int, int], None]
StepEndFn = Callable[[Step, int, int], None]


@dataclass
class RunResult:
    success: bool
    completed: int
    total: int
    failed_index: Optional[int] = None
    failed_step: Optional[Step] = None
    stopped: bool = False


class Runner:
    def __init__(
        self,
        plan: Plan,
        *,
        on_log: Optional[LogFn] = None,
        on_step_start: Optional[StepStartFn] = None,
        on_step_end: Optional[StepEndFn] = None,
        dry_run: bool = False,
    ):
        self.plan = plan
        self.steps = plan.steps
        self._log = on_log or (lambda line: print(line, flush=True))
        self._on_step_start = on_step_start
        self._on_step_end = on_step_end
        self._dry_run = dry_run

        self._stop_requested = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Ask the run to stop after terminating the command in flight."""
        self._stop_requested.set()
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        self._log("[info] Stop requested, terminating current command...")
        try:
            # Kill the whole process group: `docker run` spawns children, and
            # terminating only the parent leaves the container running.
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except OSError as exc:
                self._log(f"[warning] Could not terminate process: {exc}")

    @property
    def stopping(self) -> bool:
        return self._stop_requested.is_set()

    # ------------------------------------------------------------------
    def run(self, start_from: int = 0) -> RunResult:
        total = len(self.steps)
        if total == 0:
            self._log("[warning] Nothing to run.")
            return RunResult(success=True, completed=0, total=0)

        if start_from > 0:
            self._log(f"[info] Resuming from step {start_from + 1}/{total}")

        current_project = None
        for index in range(start_from, total):
            if self._stop_requested.is_set():
                self._log("[info] Pipeline cancelled.")
                return RunResult(False, index, total, index, self.steps[index], stopped=True)

            step = self.steps[index]
            if step.project != current_project:
                current_project = step.project
                self._log(f"\n=== Project {current_project} ===")

            if self._on_step_start:
                self._on_step_start(step, index, total)

            rc = self._execute(step, index, total)

            if self._on_step_end:
                self._on_step_end(step, index, rc)

            if rc != 0:
                if self._stop_requested.is_set():
                    self._log("[info] Pipeline cancelled.")
                    return RunResult(False, index, total, index, step, stopped=True)
                self._log(f"[error] {step.label} failed with exit code {rc}")
                self._log(
                    f"[info] Fix the problem, then resume from step {index + 1} "
                    f"with: gs-recon run ... --start-at {index + 1}"
                )
                return RunResult(False, index, total, index, step)

            self._log(f"[success] {step.label} completed.")

        self._log("\n=== Pipeline completed successfully ===")
        return RunResult(True, total, total)

    # ------------------------------------------------------------------
    def _execute(self, step: Step, index: int, total: int) -> int:
        self._log(f"\n[stage {index + 1}/{total}] {step.label}")
        self._log(f"$ {step.display()}")
        if self._dry_run:
            self._log("[dry-run] not executed")
            return 0

        try:
            proc = subprocess.Popen(
                step.exec_argv(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,   # own process group, so stop() is thorough
            )
        except OSError as exc:
            self._log(f"[error] Failed to start '{step.label}': {exc}")
            return 127

        with self._proc_lock:
            self._proc = proc

        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                self._log(line.rstrip("\n"))
        finally:
            proc.wait()
            with self._proc_lock:
                self._proc = None
        return proc.returncode or 0
