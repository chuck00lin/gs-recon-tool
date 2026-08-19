"""Stage 1 -- video frames: extract, then keep only the sharpest."""

from __future__ import annotations

import pathlib
import sys
from typing import Optional

from ..config import Config
from .base import Step

TOOL_MODULE = "gs_recon.tools.frame_extract"


def build_frame_steps(
    cfg: Config,
    project: pathlib.Path,
    video: Optional[pathlib.Path],
    *,
    python_executable: str = "",
) -> list[Step]:
    if not cfg.frames.enabled:
        return []
    if video is None:
        # An "existing project" input has no source video -- frames are already
        # on disk, so silently skip rather than failing the whole run.
        return []

    python = python_executable or sys.executable
    images_dir = project / "images"
    f = cfg.frames

    extract = [
        python, "-m", TOOL_MODULE, "extract",
        "--input", str(video),
        "--output", str(images_dir),
        "--sample-rate", str(f.sample_rate_fps),
        "--rotation", str(f.rotation),
        "--jpg-q", str(f.jpeg_quality),
        "--resolution", f.resolution,
    ]
    if f.resolution == "custom":
        extract += ["--width", str(f.custom_width)]
    if f.trim_start > 0:
        extract += ["--trim-start", str(f.trim_start)]
    if f.trim_end > 0:
        extract += ["--trim-end", str(f.trim_end)]

    filt = [
        python, "-m", TOOL_MODULE, "filter",
        "--input", str(images_dir),
        "--mode", f.filter.mode,
        "--exts", "jpg,jpeg,png",
    ]
    if f.filter.target_is_percentage():
        filt += ["--target-percentage", str(f.filter.target_value())]
    else:
        filt += ["--target-count", str(max(1, int(f.filter.target_value())))]
    if f.filter.mode == "balanced":
        filt += ["--scalar", str(max(1, f.filter.scalar))]
    elif f.filter.mode == "custom":
        filt += ["--groups", str(max(1, f.filter.groups))]

    return [
        Step(
            key="frames.extract",
            label="Frame extraction",
            stage="frames",
            argv=extract,
            project=project,
            note=f"Sample {video.name} at {f.sample_rate_fps} fps into {images_dir}",
        ),
        Step(
            key="frames.filter",
            label="Frame filtering",
            stage="frames",
            argv=filt,
            project=project,
            note=(
                f"Keep the sharpest {f.filter.target} of the extracted frames "
                f"(mode: {f.filter.mode}). Deletes the rest in place."
            ),
        ),
    ]
