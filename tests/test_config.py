import pathlib

import pytest

from gs_recon.config import Config


def test_defaults_are_valid():
    assert Config().validate() == []


def test_yaml_round_trip(tmp_path):
    cfg = Config()
    cfg.frames.sample_rate_fps = 7
    cfg.splat.prune.alpha = 0.02
    cfg.frames.filter.target = "300"

    path = cfg.save(tmp_path / "c.yaml")
    reloaded = Config.load(path)

    assert reloaded.frames.sample_rate_fps == 7
    assert reloaded.splat.prune.alpha == 0.02
    assert reloaded.to_dict() == cfg.to_dict()


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="bogus"):
        Config.from_dict({"frames": {"bogus": 1}})


def test_version_mismatch_is_rejected(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text("version: 99\n")
    with pytest.raises(ValueError, match="version"):
        Config.load(path)


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda c: setattr(c.frames, "resolution", "nope"), "resolution"),
        (lambda c: setattr(c.frames.filter, "target", "abc"), "target"),
        (lambda c: setattr(c.frames.filter, "target", "150%"), "percentage"),
        (lambda c: setattr(c.sfm, "matcher", "nope"), "matcher"),
        (lambda c: setattr(c.splat, "iterations", 0), "iterations"),
    ],
)
def test_validation_catches_bad_values(mutate, expected):
    cfg = Config()
    mutate(cfg)
    assert any(expected in problem for problem in cfg.validate())


def test_reorganize_requires_undistort():
    cfg = Config()
    cfg.sfm.reorganize = True
    assert any("undistort" in p for p in cfg.validate())


def test_conflicting_appearance_options_are_flagged():
    cfg = Config()
    cfg.splat.ppisp = True
    cfg.splat.bilateral_grid = True
    assert any("bilateral_grid" in p for p in cfg.validate())


def test_all_stages_disabled_is_flagged():
    cfg = Config()
    cfg.frames.enabled = cfg.sfm.enabled = cfg.splat.enabled = False
    assert any("nothing to run" in p for p in cfg.validate())


@pytest.mark.parametrize(
    "target, is_pct, value",
    [("20%", True, 20.0), ("300", False, 300.0), (" 12.5% ", True, 12.5)],
)
def test_filter_target_parsing(target, is_pct, value):
    cfg = Config()
    cfg.frames.filter.target = target
    assert cfg.frames.filter.target_is_percentage() is is_pct
    assert cfg.frames.filter.target_value() == value
