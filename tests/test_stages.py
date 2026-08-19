import pathlib
import shlex

import pytest

from gs_recon.config import Config
from gs_recon.stages import build_frame_steps, build_sfm_steps, build_splat_steps
from gs_recon.stages.splat import SplatConfigError

PROJECT = pathlib.Path("/data/plantA-frames")
VIDEO = pathlib.Path("/data/plantA.mp4")


def joined(step):
    return step.display()


# -- frames ------------------------------------------------------------------
def test_frame_steps_cover_extract_and_filter():
    steps = build_frame_steps(Config(), PROJECT, VIDEO)
    assert [s.key for s in steps] == ["frames.extract", "frames.filter"]


def test_percentage_and_count_targets_use_different_flags():
    cfg = Config()
    cfg.frames.filter.target = "20%"
    assert "--target-percentage" in joined(build_frame_steps(cfg, PROJECT, VIDEO)[1])

    cfg.frames.filter.target = "300"
    filter_step = build_frame_steps(cfg, PROJECT, VIDEO)[1]
    assert "--target-count 300" in joined(filter_step)
    assert "--target-percentage" not in joined(filter_step)


def test_existing_project_without_video_yields_no_frame_steps():
    assert build_frame_steps(Config(), PROJECT, None) == []


def test_custom_resolution_emits_width():
    cfg = Config()
    cfg.frames.resolution = "custom"
    cfg.frames.custom_width = 960
    assert "--width 960" in joined(build_frame_steps(cfg, PROJECT, VIDEO)[0])


def test_zero_trim_is_omitted():
    assert "--trim-start" not in joined(build_frame_steps(Config(), PROJECT, VIDEO)[0])


# -- sfm ---------------------------------------------------------------------
def test_sfm_default_pipeline_shape():
    keys = [s.key for s in build_sfm_steps(Config(), PROJECT)]
    assert keys == ["sfm.features", "sfm.match", "sfm.map", "sfm.convert"]


def test_optional_steps_appear_when_enabled():
    cfg = Config()
    cfg.sfm.undistort = cfg.sfm.reorganize = cfg.sfm.orient = True
    keys = [s.key for s in build_sfm_steps(cfg, PROJECT)]
    assert keys == [
        "sfm.features", "sfm.match", "sfm.map",
        "sfm.undistort", "sfm.reorganize", "sfm.orient", "sfm.convert",
    ]


def test_every_docker_step_runs_as_the_calling_user():
    import os

    expected = f"--user {os.getuid()}:{os.getgid()}"
    for step in build_sfm_steps(Config(), PROJECT):
        if step.argv[0] == "docker":
            assert expected in joined(step)


def test_loop_matcher_mounts_the_vocab_tree():
    command = joined(build_sfm_steps(Config(), PROJECT)[1])
    assert "/vocab_trees:ro" in command
    assert "--SequentialMatching.loop_detection 1" in command
    assert "sequential_matcher" in command   # loop closure reuses the sequential matcher


def test_exhaustive_matcher_does_not_mount_the_vocab_tree():
    cfg = Config()
    cfg.sfm.matcher = "exhaustive"
    command = joined(build_sfm_steps(cfg, PROJECT)[1])
    assert "/vocab_trees" not in command
    assert "exhaustive_matcher" in command


def test_project_is_mounted_at_ws():
    assert f"-v {PROJECT}:/ws:rw" in joined(build_sfm_steps(Config(), PROJECT)[0])


# -- splat -------------------------------------------------------------------
@pytest.fixture
def built_repo(tmp_path):
    repo = tmp_path / "LichtFeld-Studio"
    (repo / "build").mkdir(parents=True)
    (repo / "build" / "LichtFeld-Studio").write_text("#!/bin/sh\n")
    return repo


@pytest.fixture
def splat_cfg(built_repo):
    cfg = Config()
    cfg.docker.lfs_repo_host = str(built_repo)
    cfg.docker.lfs_repo_container = "/home/tester/projects/LichtFeld-Studio"
    return cfg


def test_splat_steps_shape(splat_cfg):
    keys = [s.key for s in build_splat_steps(splat_cfg, PROJECT)]
    assert keys == ["splat.train", "splat.export", "splat.prune"]


def test_training_always_uses_gut_mode(splat_cfg):
    command = joined(build_splat_steps(splat_cfg, PROJECT)[0])
    assert "--gut" in command
    # The old image's flag must never reappear -- it no longer exists upstream.
    assert "--pose-opt" not in command


def test_gpu_selection_goes_through_docker_not_the_binary(splat_cfg):
    splat_cfg.docker.gpus = "device=1"
    command = joined(build_splat_steps(splat_cfg, PROJECT)[0])
    assert "--gpus device=1" in command
    assert "--gpu " not in command   # LichtFeld has no such flag


def test_dataset_is_mounted_at_its_own_path(splat_cfg):
    command = joined(build_splat_steps(splat_cfg, PROJECT)[0])
    assert f"-v {PROJECT}:{PROJECT}:rw" in command


def test_checkout_is_mounted_at_the_runpath_location(splat_cfg, built_repo):
    command = joined(build_splat_steps(splat_cfg, PROJECT)[0])
    assert f"-v {built_repo}:/home/tester/projects/LichtFeld-Studio:rw" in command


def test_quality_flags_follow_config(splat_cfg):
    splat_cfg.splat.ppisp = False
    splat_cfg.splat.enable_mip = False
    splat_cfg.splat.bilateral_grid = True
    command = joined(build_splat_steps(splat_cfg, PROJECT)[0])
    assert "--ppisp" not in command
    assert "--enable-mip" not in command
    assert "--bilateral-grid" in command


def test_max_cap_can_be_disabled(splat_cfg):
    splat_cfg.splat.max_cap = None
    assert "--max-cap" not in joined(build_splat_steps(splat_cfg, PROJECT)[0])


def test_extra_args_are_appended(splat_cfg):
    splat_cfg.splat.extra_args = "--sh-degree 2"
    assert "--sh-degree 2" in joined(build_splat_steps(splat_cfg, PROJECT)[0])


def test_export_tolerates_a_ply_written_by_training(splat_cfg):
    # Blindly converting turns a successful run into a failed pipeline, so the
    # export step has to probe for what training actually produced.
    command = joined(build_splat_steps(splat_cfg, PROJECT)[1])
    assert "skipping convert" in command
    assert "checkpoint.resume" in command


def test_prune_can_be_disabled(splat_cfg):
    splat_cfg.splat.prune.enabled = False
    assert [s.key for s in build_splat_steps(splat_cfg, PROJECT)] == [
        "splat.train", "splat.export",
    ]


def test_missing_checkout_raises_with_guidance(tmp_path):
    cfg = Config()
    cfg.docker.lfs_repo_host = str(tmp_path / "nowhere")
    with pytest.raises(SplatConfigError, match="build/LichtFeld-Studio"):
        build_splat_steps(cfg, PROJECT)


def test_disabled_stage_produces_nothing(splat_cfg):
    splat_cfg.splat.enabled = False
    assert build_splat_steps(splat_cfg, PROJECT) == []


# -- Step primitive ----------------------------------------------------------
def test_override_is_executed_through_a_shell():
    step = build_sfm_steps(Config(), PROJECT)[0]
    step.override = "echo hello && echo world"
    assert step.exec_argv()[:2] == ["/bin/bash", "-c"]
    assert step.display() == "echo hello && echo world"


def test_paths_with_spaces_survive_quoting():
    project = pathlib.Path("/data/my plant/day 1-frames")
    command = joined(build_sfm_steps(Config(), project)[0])
    assert shlex.split(command)   # parses cleanly
    assert str(project) in shlex.split(command)[shlex.split(command).index("-v") + 1]


def test_export_name_tracks_iterations_by_default(splat_cfg):
    splat_cfg.splat.iterations = 15000
    command = joined(build_splat_steps(splat_cfg, PROJECT)[2])   # prune step
    assert "splat_15000.ply" in command
    assert "splat_30000.ply" not in command


def test_explicit_export_name_is_respected(splat_cfg):
    splat_cfg.splat.export_ply = "final.ply"
    assert "final.ply" in joined(build_splat_steps(splat_cfg, PROJECT)[2])
