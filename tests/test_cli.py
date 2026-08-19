import pytest

from gs_recon.cli import _apply_override, _parse_only
from gs_recon.config import Config


def test_override_types_are_coerced():
    cfg = Config()
    _apply_override(cfg, "splat.iterations=15000")
    _apply_override(cfg, "splat.prune.alpha=0.02")
    _apply_override(cfg, "splat.headless=false")
    _apply_override(cfg, "frames.filter.target=30%")
    _apply_override(cfg, "docker.extra_run_args=--shm-size=8g")

    assert cfg.splat.iterations == 15000
    assert cfg.splat.prune.alpha == 0.02
    assert cfg.splat.headless is False
    assert cfg.frames.filter.target == "30%"
    assert cfg.docker.extra_run_args == ["--shm-size=8g"]


@pytest.mark.parametrize(
    "bad, message",
    [
        ("splat.iterations", "KEY=VALUE"),
        ("splat.nope=1", "no such config key"),
        ("nope.thing=1", "no such config section"),
        ("splat.headless=maybe", "boolean"),
    ],
)
def test_bad_overrides_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        _apply_override(Config(), bad)


def test_parse_only():
    assert _parse_only("sfm,splat") == ["sfm", "splat"]
    assert _parse_only(None) is None
    with pytest.raises(ValueError, match="unknown stage"):
        _parse_only("sfm,bogus")
