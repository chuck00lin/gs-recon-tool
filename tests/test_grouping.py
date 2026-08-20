"""The sharpness filter's grouping arithmetic, and how it reads to a user."""

import numpy as np
import pytest

from gs_recon.config import Config
from gs_recon.tools.grouping import (
    DEFAULT_GROUP_SIZE,
    MIN_GROUP_SIZE,
    describe_selection,
    group_count_for,
)
from gs_recon.tools.image_selector import ImageSelector


# -- arithmetic --------------------------------------------------------------
def test_group_count_follows_the_frame_count():
    assert group_count_for(1000, 10) == 100
    assert group_count_for(999, 10) == 100
    assert group_count_for(37, 10) == 4


def test_group_count_is_clamped_to_something_sane():
    assert group_count_for(0, 10) == 1
    assert group_count_for(5, 500) == 1
    assert group_count_for(100, 1) == group_count_for(100, MIN_GROUP_SIZE)


def test_default_group_size_keeps_two_of_every_ten_at_the_default_target():
    cfg = Config()                     # 20% kept, groups of 10
    text = cfg.frames.filter.describe(1000)
    assert "100 groups" in text
    assert "sharpest ~2 per group" in text
    assert "⚠" not in text             # the default must not warn about itself


def test_description_warns_and_suggests_when_groups_are_too_big():
    text = describe_selection(1000, 200, 40)
    assert "⚠" in text and "try 10" in text


def test_description_warns_when_groups_are_finer_than_the_keep_rate():
    text = describe_selection(1000, 200, 3)
    assert "keep nothing" in text and "try 5" in text


def test_description_works_before_any_frame_exists():
    text = describe_selection(None, None, 10, ratio=0.2)
    assert "10" in text and "20%" in text


# -- config ------------------------------------------------------------------
def test_group_size_is_validated():
    cfg = Config()
    cfg.frames.filter.group_size = 1
    assert any("group_size" in problem for problem in cfg.validate())


def test_legacy_scalar_configs_still_load():
    """1.0.0 wrote `scalar`; rejecting it would break every saved config."""
    cfg = Config.from_dict({"frames": {"filter": {"scalar": 2, "target": "20%"}}})
    assert cfg.frames.filter.group_size == 10          # 2 kept per group at 20%
    assert "scalar" not in cfg.to_dict()["frames"]["filter"]


def test_legacy_scalar_migration_respects_the_target():
    cfg = Config.from_dict({"frames": {"filter": {"scalar": 2, "target": "35%"}}})
    assert cfg.frames.filter.group_size == 6           # ~2 kept per group at 35%


def test_an_explicit_group_size_wins_over_a_stale_scalar():
    cfg = Config.from_dict({"frames": {"filter": {"scalar": 4, "group_size": 8}}})
    assert cfg.frames.filter.group_size == 8


def test_filter_step_passes_the_group_size(tmp_path):
    from gs_recon.stages import build_frame_steps

    cfg = Config()
    cfg.frames.filter.group_size = 12
    steps = build_frame_steps(cfg, tmp_path / "p", tmp_path / "v.mp4")
    assert "--group-size 12" in steps[1].display()
    assert "--scalar" not in steps[1].display()


# -- selection ---------------------------------------------------------------
@pytest.fixture
def frames(tmp_path):
    """60 frames whose sharpness wanders, as a handheld capture's does."""
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    paths = []
    for index in range(60):
        detail = 8 + 30 * abs(np.sin(index / 9))
        image = rng.normal(128, detail, (48, 64, 3)).clip(0, 255).astype(np.uint8)
        path = tmp_path / f"frame_{index:06d}.jpg"
        cv2.imwrite(str(path), image)
        paths.append(str(path))
    return paths


def test_one_survivor_per_group_spreads_the_selection(frames):
    """Group size == the keep rate is the evenly-spaced case."""
    selected = ImageSelector(frames).filter_sharpest_images(
        12, group_count_for(len(frames), 5)
    )
    kept = sorted(int(path[-10:-4]) for path in selected)
    gaps = np.diff(kept)
    assert len(kept) == 12
    # Ideal spacing is 5. Group boundaries can put two survivors side by side,
    # but only rarely, and no hole may grow to twice the ideal.
    assert (gaps == 1).sum() <= 2
    assert gaps.max() <= 10


def test_larger_groups_trade_spacing_for_sharpness(frames):
    selector = ImageSelector(frames)
    even = selector.filter_sharpest_images(12, group_count_for(60, 5))
    clumped = selector.filter_sharpest_images(12, group_count_for(60, 20))
    sharpness = dict((path, value) for value, path in selector.image_fm)

    def adjacency(paths):
        kept = sorted(int(p[-10:-4]) for p in paths)
        return sum(1 for gap in np.diff(kept) if gap == 1)

    assert adjacency(clumped) > adjacency(even)
    assert sum(sharpness[p] for p in clumped) > sum(sharpness[p] for p in even)


def test_default_group_size_is_the_documented_one():
    assert Config().frames.filter.group_size == DEFAULT_GROUP_SIZE
