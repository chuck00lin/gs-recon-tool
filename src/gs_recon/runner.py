"""Execute a Plan, streaming output to whatever front end is watching.

Deliberately Qt-free: the CLI drives it directly and the GUI wraps it in a
QThread. One execution path means the two front ends cannot drift apart.

Stopping a run is the part that repays careful reading. A `docker run` client
forwards signals into the container, where PID 1 -- an entrypoint or a
`bash -lc` wrapper -- ignores SIGTERM by kernel rule, so "terminate the child"
alone leaves training running and the reader thread blocked on its output
forever. Two things make stop reliable: containers are started with `--init`
(see stages/base.py) so a real init process forwards the signal to the actual
workload, and every `docker run` gets a name here so the container itself can
be killed if the signal is ignored anyway.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from .pipeline import Plan
from .stages import Step

LogFn = Callable[[str], None]
StepStartFn = Callable[[Step, int, int], None]
StepEndFn = Callable[[Step, int, int], None]

# How long a command gets to exit on SIGTERM before the container is killed
# outright. Long enough for LichtFeld to flush a checkpoint, short enough that
# a stuck process does not hold the UI hostage.
GRACE_SECONDS = 10.0
KILL_SECONDS = 5.0


@dataclass
class RunResult:
    success: bool
    completed: int
    total: int
    failed_index: Optional[int] = None
    failed_step: Optional[Step] = None
    stopped: bool = False


def name_docker_run(argv: list[str], name: str) -> tuple[list[str], Optional[str]]:
    """Give a ``docker run`` command a name, so the container can be killed.

    Anything that is not a docker run (host-side python steps, the ``override``
    escape hatch, which goes through a shell) is returned untouched.
    """
    if len(argv) >= 2 and os.path.basename(argv[0]) == "docker" and argv[1] == "run":
        return [*argv[:2], "--name", name, *argv[2:]], name
    return argv, None


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
        self._container: Optional[str] = None
        self._proc_lock = threading.Lock()
        # Unique per Runner so a resumed run never collides with a container
        # left behind by the attempt before it.
        self._token = uuid.uuid4().hex[:6]

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Ask the run to stop, and make sure the command in flight really dies."""
        already_stopping = self._stop_requested.is_set()
        self._stop_requested.set()
        with self._proc_lock:
            proc, container = self._proc, self._container
        if proc is None or proc.poll() is not None:
            return
        if already_stopping:
            # A second stop means the first one is taking too long: skip the
            # polite phase entirely.
            grace = 0.0
        else:
            self._log("[info] Stop requested, terminating the current command...")
            self._signal(proc, signal.SIGTERM)
            grace = GRACE_SECONDS
        # Always escalate on a worker thread: stop() is called from the GUI
        # thread, and waiting for a stubborn container there would freeze the
        # very window the user is trying to stop from.
        threading.Thread(
            target=self._force_stop, args=(proc, container, grace), daemon=True
        ).start()

    @property
    def stopping(self) -> bool:
        return self._stop_requested.is_set()

    # ------------------------------------------------------------------
    def _signal(self, proc: subprocess.Popen, sig: int) -> None:
        try:
            # The whole process group: `docker run` spawns children, and
            # signalling only the parent leaves them behind.
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.send_signal(sig)
            except OSError as exc:
                self._log(f"[warning] Could not signal the process: {exc}")

    def _force_stop(
        self,
        proc: subprocess.Popen,
        container: Optional[str],
        grace: float = GRACE_SECONDS,
    ) -> None:
        """Escalate until the command is actually gone."""
        if self._wait(proc, grace):
            return
        if container:
            self._log(f"[info] Still running -- killing container {container}.")
            try:
                subprocess.run(
                    ["docker", "kill", container],
                    capture_output=True, text=True, timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                self._log(f"[warning] docker kill failed: {exc}")
            if self._wait(proc, KILL_SECONDS):
                return
        self._log("[warning] Force killing the process group.")
        self._signal(proc, signal.SIGKILL)

    @staticmethod
    def _wait(proc: subprocess.Popen, seconds: float) -> bool:
        if seconds <= 0:
            return proc.poll() is not None
        try:
            proc.wait(timeout=seconds)
            return True
        except subprocess.TimeoutExpired:
            return False

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

        argv, container = name_docker_run(
            step.exec_argv(), f"gsr-{self._token}-{index + 1}"
        )
        try:
            proc = subprocess.Popen(
                argv,
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
            self._container = container

        # A stop that arrived while the process was starting would have found
        # no process to signal, so honour it now.
        if self._stop_requested.is_set():
            self.stop()

        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                self._log(line.rstrip("\n"))
        finally:
            proc.wait()
            with self._proc_lock:
                self._proc = None
                self._container = None
        return proc.returncode or 0
