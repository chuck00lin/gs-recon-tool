"""Frame extraction: the properties SfM depends on.

Filename order has to equal capture order, one run's output has to be the only
thing in the folder, and the requested sample rate has to be the rate you get.
Each of those was broken in 1.0.0 and each has a test here.
"""

import pathlib

import cv2
import numpy as np
import pytest

from gs_recon.tools.frame_extract import (
    FrameSchedule,
    cli_extract,
    clear_previous_frames,
    frame_name,
)


# -- naming ------------------------------------------------------------------
def test_names_sort_in_capture_order_past_9999():
    names = [frame_name(i) for i in range(10_005)]
    assert sorted(names) == names


# -- schedule ----------------------------------------------------------------
def _kept(rate, timestamps, tolerance):
    schedule = FrameSchedule(rate, tolerance_ms=tolerance)
    return [t for t in timestamps if schedule.wants(t)]


def test_half_rate_keeps_every_second_frame():
    # 30 fps source, whole-millisecond timestamps as containers report them.
    timestamps = [round(i * 1000 / 30) for i in range(60)]
    kept = _kept(15, timestamps, tolerance=1000 / 30 / 2)
    assert kept == timestamps[::2]


def test_quantised_timestamps_do_not_drift():
    """Without the tolerance this alternates 1-2-1-2 and over-samples."""
    timestamps = [round(i * 1000 / 30) for i in range(300)]
    kept = _kept(15, timestamps, tolerance=1000 / 30 / 2)
    assert 149 <= len(kept) <= 151


def test_non_integer_ratio_lands_on_the_requested_rate():
    # 29.97 fps for 10 s; the old "every Nth frame" rounded N to 1 and kept all.
    timestamps = [i * 1000 / 29.97 for i in range(300)]
    kept = _kept(20, timestamps, tolerance=1000 / 29.97 / 2)
    assert 199 <= len(kept) <= 201


def test_a_gap_does_not_trigger_a_catch_up_burst():
    # A VFR clip that stalls for a second, then resumes at 60 fps.
    timestamps = [0.0, 1000.0] + [1000.0 + i * 1000 / 60 for i in range(1, 60)]
    kept = _kept(6, timestamps, tolerance=8.0)
    gaps = np.diff(kept)
    assert gaps.min() > 100     # never two frames back to back
    assert kept[:2] == [0.0, 1000.0]


# -- stale output ------------------------------------------------------------
def test_clear_previous_frames_removes_only_our_own(tmp_path):
    for i in range(3):
        (tmp_path / frame_name(i)).touch()
    (tmp_path / "frame_0007.jpg").touch()        # 1.0.0's 4-digit naming
    (tmp_path / "reference.png").touch()
    (tmp_path / "notes.txt").touch()

    removed, foreign = clear_previous_frames(str(tmp_path))

    assert removed == 4
    assert foreign == 1
    assert {p.name for p in tmp_path.iterdir()} == {"reference.png", "notes.txt"}


# -- end to end --------------------------------------------------------------
@pytest.fixture
def clip(tmp_path):
    """A 6 s, 30 fps clip whose frames are numbered by brightness."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("no mp4v encoder in this OpenCV build")
    for index in range(180):
        writer.write(np.full((48, 64, 3), index, np.uint8))
    writer.release()
    return path


def _frames(folder: pathlib.Path) -> list[str]:
    return sorted(p.name for p in folder.glob("*.jpg"))


def _extract(clip, out, **kwargs):
    args = dict(
        sample_rate=15, rotation=0, jpg_q=90, resolution="original",
        custom_width=1280, trim_start=0, trim_end=0,
    )
    args.update(kwargs)
    return cli_extract(str(clip), str(out), **args)


def test_extraction_hits_the_requested_rate(clip, tmp_path):
    out = tmp_path / "images"
    assert _extract(clip, out, sample_rate=10) == 0
    # 6 s at 10 fps, allowing one frame either side for rounding at the ends.
    assert 59 <= len(_frames(out)) <= 61


def test_rerun_replaces_the_previous_extraction(clip, tmp_path):
    out = tmp_path / "images"
    _extract(clip, out, sample_rate=15)
    dense = _frames(out)
    assert len(dense) > 80

    _extract(clip, out, sample_rate=3)
    sparse = _frames(out)

    assert len(sparse) < 25
    # The old run must be gone: leaving it behind is what made the sequence
    # jump backwards partway through.
    assert sparse == [frame_name(i) for i in range(len(sparse))]
    assert set(sparse).issubset(set(dense))


def test_keep_existing_opts_out(clip, tmp_path):
    out = tmp_path / "images"
    _extract(clip, out, sample_rate=15)
    before = len(_frames(out))

    _extract(clip, out, sample_rate=3, keep_existing=True)

    assert len(_frames(out)) == before


def test_trim_drops_the_requested_seconds(clip, tmp_path):
    out = tmp_path / "images"
    assert _extract(clip, out, sample_rate=10, trim_start=2.0, trim_end=1.0) == 0
    frames = _frames(out)
    assert 29 <= len(frames) <= 31          # 3 s of the 6 s clip at 10 fps
    # Frame 60 is the 2 s mark; the tolerance covers the lossy round trip
    # through the encoder, not seek slop (seeking is frame-exact).
    first = cv2.imread(str(out / frames[0]))
    assert abs(int(first[0, 0, 0]) - 60) <= 6
