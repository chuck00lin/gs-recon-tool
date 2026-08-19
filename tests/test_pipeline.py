import pathlib

import pytest

from gs_recon.config import Config
from gs_recon.pipeline import build_plan, discover_inputs, project_dir_for_video


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "day01").mkdir()
    (tmp_path / "day02").mkdir()
    (tmp_path / "day01" / "a.mp4").touch()
    (tmp_path / "day01" / "b.MOV").touch()
    (tmp_path / "day02" / "c.mp4").touch()
    (tmp_path / "notes.txt").touch()
    return tmp_path


def test_video_file_becomes_a_frames_project(tree):
    inputs, warnings = discover_inputs([tree / "day01" / "a.mp4"])
    assert warnings == []
    assert len(inputs) == 1
    assert inputs[0].project.name == "a-frames"
    assert inputs[0].video is not None


def test_folder_of_videos(tree):
    inputs, _ = discover_inputs([tree / "day01"])
    assert sorted(i.project.name for i in inputs) == ["a-frames", "b-frames"]


def test_recursive_scan(tree):
    inputs, _ = discover_inputs([tree], recursive=True)
    assert len(inputs) == 3


def test_non_recursive_parent_folder_warns(tree):
    inputs, warnings = discover_inputs([tree])
    assert inputs == []
    assert any("--recursive" in w for w in warnings)


def test_existing_project_folder(tmp_path):
    project = tmp_path / "plantA-frames"
    (project / "images").mkdir(parents=True)
    inputs, warnings = discover_inputs([project])
    assert warnings == []
    assert inputs[0].video is None
    assert inputs[0].project == project.resolve()


def test_missing_path_warns_instead_of_raising(tmp_path):
    inputs, warnings = discover_inputs([tmp_path / "nope.mp4"])
    assert inputs == []
    assert "does not exist" in warnings[0]


def test_non_video_file_warns(tree):
    _, warnings = discover_inputs([tree / "notes.txt"])
    assert any("not a recognised video" in w for w in warnings)


def test_duplicates_are_collapsed(tree):
    video = tree / "day01" / "a.mp4"
    inputs, _ = discover_inputs([video, video, tree / "day01"])
    assert sorted(i.project.name for i in inputs) == ["a-frames", "b-frames"]


def test_only_restricts_stages_without_mutating_config(tree):
    cfg = Config()
    inputs, _ = discover_inputs([tree / "day01" / "a.mp4"])
    plan = build_plan(cfg, inputs, only=["sfm"])

    assert {step.stage for step in plan.steps} == {"sfm"}
    assert cfg.frames.enabled is True    # caller's config untouched


def test_unknown_stage_is_rejected(tree):
    inputs, _ = discover_inputs([tree / "day01" / "a.mp4"])
    with pytest.raises(ValueError, match="unknown stage"):
        build_plan(Config(), inputs, only=["bogus"])


def test_existing_project_skips_frame_steps(tmp_path):
    project = tmp_path / "p-frames"
    (project / "images").mkdir(parents=True)
    inputs, _ = discover_inputs([project])
    plan = build_plan(Config(), inputs, only=["frames", "sfm"])
    assert {step.stage for step in plan.steps} == {"sfm"}


def test_project_dir_naming():
    assert project_dir_for_video(pathlib.Path("/d/clip.mp4")).name == "clip-frames"
