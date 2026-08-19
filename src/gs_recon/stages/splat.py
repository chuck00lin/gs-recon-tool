"""Stage 3 -- 3D Gaussian Splatting with LichtFeld Studio.

Two facts drive the shape of these commands, and both are easy to get wrong:

1. The ``lichtfeld-studio`` image contains **no** LichtFeld binary. The
   executable lives in the host checkout's ``build/`` directory and is produced
   by building inside the container against a bind mount.
2. That binary's RUNPATH is baked to ``/home/<builder>/projects/LichtFeld-Studio``
   (it points at ``build/vcpkg_installed/x64-linux/lib``). Mount the checkout
   anywhere else and it dies with a missing-shared-library error.

So we mount the checkout at exactly its build-time path -- derived by probing
the image for the user it was built for -- and mount the dataset at its own
host path so no translation is ever needed.
"""

from __future__ import annotations

import pathlib
import shlex
from typing import Optional

from ..config import Config
from .base import Step, docker_base

# The binary writes its shader/pipeline cache here and expects it to exist.
# Every `docker run --rm` gets a fresh /tmp, so this runs each time.
CACHE_BOOTSTRAP = (
    "mkdir -p /tmp/LichtFeldStudio/cache && "
    "chmod -R 777 /tmp/LichtFeldStudio 2>/dev/null || true"
)


class SplatConfigError(RuntimeError):
    """Raised when the LichtFeld checkout cannot be located."""


def _lfs_docker_head(cfg: Config, project: pathlib.Path) -> tuple[list[str], str]:
    host_repo = cfg.docker.resolved_lfs_repo_host()
    if host_repo is None:
        raise SplatConfigError(
            "Could not locate a LichtFeld-Studio checkout. Set docker.lfs_repo_host "
            "in your config (or the LICHTFELD_STUDIO_ROOT environment variable) to "
            "the folder that contains build/LichtFeld-Studio. Run `gs-recon doctor` "
            "for details."
        )
    if not (host_repo / "build" / "LichtFeld-Studio").is_file():
        raise SplatConfigError(
            f"{host_repo} has no build/LichtFeld-Studio binary. Build it first:\n"
            f"    cd {host_repo} && ./docker/run_docker.sh -bu\n"
            f"then run the project's build steps inside the container."
        )

    container_repo = cfg.docker.resolved_lfs_repo_container()
    argv = docker_base(
        cfg.docker.lfs_image,
        [
            (host_repo, container_repo, "rw"),
            # Identical host/container path: no translation, works for any
            # dataset location including external drives.
            (project, str(project), "rw"),
        ],
        gpus=cfg.docker.gpus,
        ipc_host=True,
        env_vars={"NVIDIA_DRIVER_CAPABILITIES": "all"},
        extra_args=cfg.docker.extra_run_args,
    )
    return argv, container_repo


def build_splat_steps(cfg: Config, project: pathlib.Path) -> list[Step]:
    if not cfg.splat.enabled:
        return []

    s = cfg.splat
    output_dir = project / "gs"
    ply_path = output_dir / s.export_ply_name()
    checkpoint = output_dir / "checkpoints" / "checkpoint.resume"

    head, container_repo = _lfs_docker_head(cfg, project)
    steps: list[Step] = []

    # -- training -------------------------------------------------------------
    train = [
        "./build/LichtFeld-Studio",
        "-d", str(project),
        "-o", str(output_dir),
        "-i", str(s.iterations),
        "--gut",
    ]
    if s.headless:
        train.append("--headless")
    if s.max_cap:
        train += ["--max-cap", str(s.max_cap)]
    if s.ppisp:
        train.append("--ppisp")
    if s.enable_mip:
        train.append("--enable-mip")
    if s.bilateral_grid:
        train.append("--bilateral-grid")
    if s.undistort:
        train.append("--undistort")
    train += _split_extra_args(s.extra_args)

    inner_train = (
        f"{CACHE_BOOTSTRAP} && cd {shlex.quote(container_repo)} && "
        + shlex.join(train)
    )
    steps.append(Step(
        key="splat.train",
        label="Gaussian splat training",
        stage="splat",
        project=project,
        note=f"{s.iterations} iterations in GUT mode, writing to {output_dir}",
        argv=head + ["bash", "-lc", inner_train],
    ))

    # -- checkpoint -> PLY ----------------------------------------------------
    # Training may emit a PLY directly or only a resumable checkpoint depending
    # on flags, so probe rather than assume; blindly converting turns a
    # successful run into a failed pipeline.
    convert = shlex.join([
        "./build/LichtFeld-Studio", "convert",
        str(checkpoint), str(ply_path), "-y",
    ])
    inner_convert = (
        f"cd {shlex.quote(container_repo)} && "
        f"if [ -f {shlex.quote(str(ply_path))} ]; then "
        f"echo 'PLY already written by training, skipping convert'; "
        f"elif [ -f {shlex.quote(str(checkpoint))} ]; then {convert}; "
        f"else echo 'ERROR: neither {ply_path} nor {checkpoint} exists' >&2; exit 1; fi"
    )
    steps.append(Step(
        key="splat.export",
        label="Export PLY",
        stage="splat",
        project=project,
        note=f"Ensure {ply_path.name} exists, converting from the checkpoint if needed",
        argv=head + ["bash", "-lc", inner_convert],
    ))

    # -- host-side pruning ----------------------------------------------------
    if s.prune.enabled:
        import sys
        steps.append(Step(
            key="splat.prune",
            label="Prune PLY (alpha + radius)",
            stage="splat",
            project=project,
            note=(
                f"Drop Gaussians with alpha < {s.prune.alpha} and outliers beyond "
                f"p{s.prune.percentile} x {s.prune.margin} of the centre"
            ),
            argv=[
                sys.executable, "-m", "gs_recon.tools.prune_ply",
                str(ply_path), str(ply_path),
                "--alpha", str(s.prune.alpha),
                "--percentile", str(s.prune.percentile),
                "--margin", str(s.prune.margin),
            ],
        ))

    return steps


def _split_extra_args(raw: Optional[str]) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        return shlex.split(text)
    except ValueError:
        # Unbalanced quotes: pass through untouched rather than silently
        # dropping the user's intent.
        return [text]
