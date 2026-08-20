"""The Step primitive and shared docker plumbing."""

from __future__ import annotations

import pathlib
import shlex
from dataclasses import dataclass, field
from typing import Optional

from .. import env


@dataclass
class Step:
    """One executable unit of work.

    ``argv`` is the canonical form -- it is executed without a shell, so paths
    with spaces are safe. ``override`` exists purely so the GUI can offer an
    "edit the command" escape hatch for the odd case the presets do not cover;
    when set it is run through ``bash -c`` verbatim.
    """

    key: str                     # stable identifier, e.g. "sfm.match"
    label: str                   # shown in progress UI
    stage: str                   # "frames" | "sfm" | "splat"
    argv: list[str] = field(default_factory=list)
    project: Optional[pathlib.Path] = None
    note: str = ""               # one-line explanation, shown as a tooltip
    override: Optional[str] = None

    def display(self) -> str:
        if self.override is not None:
            return self.override
        return shlex.join(self.argv)

    def exec_argv(self) -> list[str]:
        if self.override is not None:
            return ["/bin/bash", "-c", self.override]
        return list(self.argv)


def docker_base(
    image: str,
    mounts: list[tuple[pathlib.Path | str, str, str]],
    *,
    gpus: str = "",
    extra_args: Optional[list[str]] = None,
    env_vars: Optional[dict[str, str]] = None,
    ipc_host: bool = False,
) -> list[str]:
    """Build the invariant head of a ``docker run --rm`` command.

    Runs as the calling user so output files are owned by them rather than root
    -- the single most common source of "I can't delete my own results".

    ``--init`` is not cosmetic: without it the container's PID 1 is a shell or
    an entrypoint script, and the kernel does not apply default signal actions
    to PID 1, so a stop request is silently ignored and training runs on. With
    it, a real init forwards SIGTERM to the workload and the container exits in
    well under a second.
    """
    argv = ["docker", "run", "--rm", "--init", "--user", f"{env.uid()}:{env.gid()}"]
    if gpus:
        argv += ["--gpus", gpus]
    if ipc_host:
        argv += ["--ipc=host"]
    for key, value in (env_vars or {}).items():
        argv += ["-e", f"{key}={value}"]
    for host, container, mode in mounts:
        argv += ["-v", f"{host}:{container}:{mode}"]
    argv += list(extra_args or [])
    argv.append(image)
    return argv
