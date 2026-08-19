"""Stage 2 -- Structure from Motion with COLMAP / GLOMAP in Docker.

The project directory is bind-mounted at ``/ws`` so every command below is
path-agnostic, and the shared vocabulary tree is mounted read-only from a
single location instead of being copied per project.
"""

from __future__ import annotations

import pathlib
import shlex
import textwrap

from .. import env
from ..config import Config
from .base import Step, docker_base

WS = "/ws"
VOCAB_MOUNT = "/vocab_trees"


def _base(cfg: Config, project: pathlib.Path, *, gpu: bool, extra_mounts=()) -> list[str]:
    mounts = [(project, WS, "rw"), *extra_mounts]
    return docker_base(
        cfg.docker.colmap_image,
        mounts,
        gpus=cfg.docker.gpus if gpu else "",
        extra_args=cfg.docker.extra_run_args,
    )


def build_sfm_steps(cfg: Config, project: pathlib.Path) -> list[Step]:
    if not cfg.sfm.enabled:
        return []

    s = cfg.sfm
    steps: list[Step] = []

    # -- 1. feature extraction ------------------------------------------------
    steps.append(Step(
        key="sfm.features",
        label="Feature extraction",
        stage="sfm",
        project=project,
        note=f"SIFT features, camera model {s.camera_model}, single shared intrinsics",
        argv=_base(cfg, project, gpu=True) + [
            "colmap", "feature_extractor",
            "--image_path", f"{WS}/images",
            "--database_path", f"{WS}/database.db",
            "--ImageReader.camera_model", s.camera_model,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1",
        ],
    ))

    # -- 2. matching ----------------------------------------------------------
    steps.append(_matching_step(cfg, project))

    # -- 3. mapping -----------------------------------------------------------
    inner = (
        f"mkdir -p {WS}/sparse && "
        f"{s.mapper} mapper "
        f"--database_path {WS}/database.db "
        f"--image_path {WS}/images "
        f"--output_path {WS}/sparse"
    )
    steps.append(Step(
        key="sfm.map",
        label=f"SfM mapping ({s.mapper})",
        stage="sfm",
        project=project,
        note="Solve camera poses and a sparse point cloud into sparse/0/",
        argv=_base(cfg, project, gpu=True) + ["bash", "-lc", inner],
    ))

    # -- 4. optional undistortion + reorganisation ----------------------------
    if s.undistort:
        steps.append(Step(
            key="sfm.undistort",
            label="Image undistortion",
            stage="sfm",
            project=project,
            note="Only needed when the downstream consumer cannot handle lens distortion",
            argv=_base(cfg, project, gpu=False) + [
                "colmap", "image_undistorter",
                "--image_path", f"{WS}/images",
                "--input_path", f"{WS}/sparse/0",
                "--output_path", f"{WS}/dense",
                "--output_type", "COLMAP",
            ],
        ))
    if s.reorganize:
        steps.append(Step(
            key="sfm.reorganize",
            label="Reorganise undistorted output",
            stage="sfm",
            project=project,
            note="Promote dense/ results to images/ + sparse/, keeping timestamped backups",
            argv=["/bin/bash", "-e", "-c", _reorganize_script(project)],
        ))

    # -- 5. optional orientation alignment ------------------------------------
    if s.orient:
        steps.append(Step(
            key="sfm.orient",
            label="Model orientation alignment",
            stage="sfm",
            project=project,
            note="Rotate the sparse model so its dominant plane is axis-aligned",
            argv=_base(cfg, project, gpu=False) + [
                "colmap", "model_orientation_aligner",
                "--image_path", f"{WS}/images",
                "--input_path", f"{WS}/sparse/0",
                "--output_path", f"{WS}/sparse/0",
            ],
        ))

    # -- 6. format conversion -------------------------------------------------
    steps.append(_convert_step(cfg, project))
    return steps


def _matching_step(cfg: Config, project: pathlib.Path) -> Step:
    s = cfg.sfm
    matcher = s.matcher
    extra_mounts = []
    colmap_matcher = matcher
    note = f"{matcher} matching"

    if matcher == "sequential-loop":
        # Reuses COLMAP's sequential matcher plus vocabulary-tree loop closure.
        colmap_matcher = "sequential"
        extra_mounts.append((env.vocab_tree_path().parent, VOCAB_MOUNT, "ro"))
        note = "Sequential matching with vocabulary-tree loop closure (best for orbit captures)"
    elif matcher == "vocab_tree":
        extra_mounts.append((env.vocab_tree_path().parent, VOCAB_MOUNT, "ro"))

    argv = _base(cfg, project, gpu=True, extra_mounts=extra_mounts) + [
        "colmap", f"{colmap_matcher}_matcher",
        "--database_path", f"{WS}/database.db",
        "--SiftMatching.use_gpu", "1",
    ]

    if matcher in {"sequential", "sequential-loop"}:
        argv += [
            "--SiftMatching.cross_check", "1",
            "--SiftMatching.guided_matching", "1",
        ]

    if matcher == "sequential-loop":
        # Tuned for elongated subjects captured in multiple orbits: a wider
        # temporal window plus more loop candidates lets the second pass anchor
        # to the first even where features are sparse.
        argv += [
            "--SequentialMatching.loop_detection", "1",
            "--SequentialMatching.vocab_tree_path",
            f"{VOCAB_MOUNT}/{env.VOCAB_TREE_NAME}",
            "--SequentialMatching.overlap", str(s.loop_overlap),
            "--SequentialMatching.loop_detection_num_images", str(s.loop_detection_num_images),
            "--SequentialMatching.loop_detection_period", str(s.loop_detection_period),
            "--SiftMatching.max_ratio", "0.7",
            "--SiftMatching.max_distance", "0.6",
        ]
    elif matcher == "vocab_tree":
        argv += ["--VocabTreeMatching.vocab_tree_path", f"{VOCAB_MOUNT}/{env.VOCAB_TREE_NAME}"]

    return Step(
        key="sfm.match",
        label="Feature matching",
        stage="sfm",
        project=project,
        note=note,
        argv=argv,
    )


def _convert_step(cfg: Config, project: pathlib.Path) -> Step:
    output_type = cfg.sfm.convert
    if output_type == "TXT+PLY":
        inner = (
            "set -euo pipefail && "
            f"colmap model_converter --input_path {WS}/sparse/0 "
            f"--output_path {WS}/sparse/0/points.ply --output_type PLY && "
            f"colmap model_converter --input_path {WS}/sparse/0 "
            f"--output_path {WS}/sparse/0 --output_type TXT"
        )
        argv = _base(cfg, project, gpu=False) + ["bash", "-lc", inner]
    else:
        if output_type == "PLY":
            out = f"{WS}/sparse/0/points.ply"
        elif output_type == "TXT":
            out = f"{WS}/sparse/0"
        else:
            out = f"{WS}/sparse/0/model.{output_type.lower()}"
        argv = _base(cfg, project, gpu=False) + [
            "colmap", "model_converter",
            "--input_path", f"{WS}/sparse/0",
            "--output_path", out,
            "--output_type", output_type,
        ]
    return Step(
        key="sfm.convert",
        label=f"Format conversion ({output_type})",
        stage="sfm",
        project=project,
        note="Write human-readable TXT and/or a PLY preview of the sparse cloud",
        argv=argv,
    )


def _reorganize_script(project: pathlib.Path) -> str:
    p = shlex.quote(str(project))
    return textwrap.dedent(f"""
        cd {p} || exit 1
        echo '=== Reorganising undistorted results ==='

        if [ ! -d dense/images ] || [ ! -d dense/sparse ]; then
            echo 'ERROR: dense/images or dense/sparse not found. Undistortion may have failed.' >&2
            exit 1
        fi

        ts=$(date +%Y%m%d_%H%M%S)
        [ -d images ] && mv images "images_original_$ts"
        [ -d sparse ] && mv sparse "sparse_original_$ts"

        echo 'Moving undistorted images...'
        mv dense/images images
        echo 'Moving undistorted sparse model...'
        mv dense/sparse sparse

        echo 'Ensuring proper sparse/0/ structure...'
        mkdir -p sparse/0
        mv sparse/*.bin sparse/0/ 2>/dev/null || true

        echo 'File reorganisation completed successfully'
    """).strip()
