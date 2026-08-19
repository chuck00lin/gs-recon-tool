"""Turn user inputs + a Config into a flat, inspectable list of steps.

Planning is deliberately separate from running: `gs-recon plan` prints exactly
what `gs-recon run` will execute, and the GUI shows the same list before you
commit a machine to eight hours of work.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .config import Config
from .stages import Step, build_frame_steps, build_sfm_steps, build_splat_steps

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
# Any one of these means "this folder is already a reconstruction project"
PROJECT_MARKERS = ("images", "database.db", "sparse", "gs")

STAGE_ORDER = ("frames", "sfm", "splat")


@dataclass
class ProjectInput:
    """One reconstruction target: a folder, plus the video that seeds it."""

    project: pathlib.Path
    video: Optional[pathlib.Path] = None

    @property
    def name(self) -> str:
        return self.project.name


@dataclass
class ProjectPlan:
    project: pathlib.Path
    video: Optional[pathlib.Path]
    steps: list[Step] = field(default_factory=list)


@dataclass
class Plan:
    projects: list[ProjectPlan] = field(default_factory=list)

    @property
    def steps(self) -> list[Step]:
        return [step for project in self.projects for step in project.steps]

    def __len__(self) -> int:
        return len(self.steps)

    def describe(self) -> str:
        lines: list[str] = []
        index = 1
        for plan in self.projects:
            source = f"  <- {plan.video}" if plan.video else "  (existing project)"
            lines.append(f"\n=== {plan.project}{source}")
            for step in plan.steps:
                lines.append(f"  [{index:>2}] {step.label}")
                if step.note:
                    lines.append(f"       # {step.note}")
                lines.append(f"       $ {step.display()}")
                index += 1
        if not lines:
            return "(nothing to do -- every stage is disabled or no input matched)"
        return "\n".join(lines).lstrip("\n")


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------
def looks_like_project(path: pathlib.Path) -> bool:
    return any((path / marker).exists() for marker in PROJECT_MARKERS)


def project_dir_for_video(video: pathlib.Path) -> pathlib.Path:
    """``/data/plant01.mp4`` -> ``/data/plant01-frames``."""
    return video.parent / f"{video.stem}-frames"


def discover_inputs(
    paths: Iterable[str | pathlib.Path],
    *,
    recursive: bool = False,
) -> tuple[list[ProjectInput], list[str]]:
    """Resolve CLI/GUI arguments into concrete reconstruction targets.

    Accepts, in any mix: video files, a folder of videos, an existing project
    folder, or (with ``recursive``) a folder whose immediate subfolders each
    hold videos. Returns the targets plus a list of warnings for anything that
    could not be interpreted, so callers can surface them instead of silently
    dropping input.
    """
    inputs: list[ProjectInput] = []
    warnings: list[str] = []
    seen: set[pathlib.Path] = set()

    def add(project: pathlib.Path, video: Optional[pathlib.Path]) -> None:
        project = project.resolve()
        if project in seen:
            return
        seen.add(project)
        inputs.append(ProjectInput(project=project, video=video))

    for raw in paths:
        path = pathlib.Path(raw).expanduser()
        if not path.exists():
            warnings.append(f"{path}: does not exist, skipped")
            continue
        path = path.resolve()

        if path.is_file():
            if path.suffix.lower() not in VIDEO_EXTS:
                warnings.append(f"{path}: not a recognised video ({sorted(VIDEO_EXTS)}), skipped")
                continue
            add(project_dir_for_video(path), path)
            continue

        videos = sorted(
            child for child in path.iterdir()
            if child.is_file() and child.suffix.lower() in VIDEO_EXTS
        )
        if videos:
            for video in videos:
                add(project_dir_for_video(video), video)
            continue

        if recursive:
            found_any = False
            for sub in sorted(child for child in path.iterdir() if child.is_dir()):
                sub_videos = sorted(
                    child for child in sub.iterdir()
                    if child.is_file() and child.suffix.lower() in VIDEO_EXTS
                )
                for video in sub_videos:
                    add(project_dir_for_video(video), video)
                    found_any = True
            if found_any:
                continue

        if looks_like_project(path):
            add(path, None)
            continue

        warnings.append(
            f"{path}: no videos and no project markers "
            f"({', '.join(PROJECT_MARKERS)}) -- skipped"
            + ("" if recursive else "; pass --recursive to scan subfolders")
        )

    return inputs, warnings


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def build_plan(
    cfg: Config,
    inputs: Iterable[ProjectInput],
    *,
    only: Optional[Iterable[str]] = None,
) -> Plan:
    """Expand every input into its steps.

    ``only`` restricts the run to a subset of stage names without mutating the
    caller's config, so ``--only sfm`` never rewrites the user's YAML.
    """
    effective = cfg
    if only is not None:
        wanted = set(only)
        unknown = wanted - set(STAGE_ORDER)
        if unknown:
            raise ValueError(
                f"unknown stage(s): {', '.join(sorted(unknown))}. "
                f"Valid stages: {', '.join(STAGE_ORDER)}"
            )
        effective = cfg.copy()
        effective.frames.enabled = "frames" in wanted
        effective.sfm.enabled = "sfm" in wanted
        effective.splat.enabled = "splat" in wanted

    plan = Plan()
    for item in inputs:
        steps: list[Step] = []
        steps += build_frame_steps(effective, item.project, item.video)
        steps += build_sfm_steps(effective, item.project)
        steps += build_splat_steps(effective, item.project)
        if steps:
            plan.projects.append(
                ProjectPlan(project=item.project, video=item.video, steps=steps)
            )
    return plan


def ensure_project_dirs(plan: Plan) -> None:
    """Create the folders the first step writes into.

    Done up front so a permission problem surfaces before a two-hour job rather
    than at the end of one.
    """
    for project_plan in plan.projects:
        project_plan.project.mkdir(parents=True, exist_ok=True)
        if any(step.stage == "frames" for step in project_plan.steps):
            (project_plan.project / "images").mkdir(parents=True, exist_ok=True)
