"""Command builders. Pure functions: config + project path -> Step list.

No Qt, no side effects, no subprocesses. This is what lets the GUI and the CLI
produce byte-identical commands, and what makes the whole plan printable with
`gs-recon plan` before anything runs.
"""

from .base import Step, docker_base
from .frames import build_frame_steps
from .sfm import build_sfm_steps
from .splat import build_splat_steps

__all__ = [
    "Step",
    "docker_base",
    "build_frame_steps",
    "build_sfm_steps",
    "build_splat_steps",
]
