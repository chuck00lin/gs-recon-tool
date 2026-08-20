"""Extract video frames and cull them down to the sharpest subset.

Usable standalone::

    python -m gs_recon.tools.frame_extract extract --input clip.mp4 --output ./images
    python -m gs_recon.tools.frame_extract filter  --input ./images --target-percentage 20

Two properties matter more than anything else here, because everything
downstream (COLMAP's sequential matcher above all) reads the output folder as
"filename order == capture order":

* the folder must contain exactly one run's worth of frames, and
* the names must sort in capture order for any frame count.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import pathlib
import shutil
import sys

import cv2

from .grouping import group_count_for
from .image_selector import ImageSelector

ROTATIONS = {
    -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
}
SCALE_DIVISORS = {"original": 1, "half": 2, "quarter": 4, "eighth": 8}

# Six digits, not four: a 4-digit name sorts wrongly the moment a run passes
# 9999 frames (``frame_10000`` lands between ``frame_1000`` and ``frame_1001``),
# which silently scrambles capture order for every consumer downstream.
FRAME_NAME = "frame_{:06d}.jpg"
FRAME_PATTERN = "frame_*.jpg"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def frame_name(index: int) -> str:
    return FRAME_NAME.format(index)


def clear_previous_frames(folder: str) -> tuple[int, int]:
    """Delete frames left by an earlier extraction into the same folder.

    Without this, a second run at a lower sample rate overwrites only the first
    N names and leaves the rest of the previous run in place -- the folder then
    holds two interleaved captures and the sequence jumps backwards partway
    through. Returns (removed, foreign) where ``foreign`` counts images that do
    not follow our naming scheme and were therefore left alone.
    """
    removed = 0
    for path in glob.glob(os.path.join(folder, FRAME_PATTERN)):
        try:
            os.remove(path)
            removed += 1
        except OSError as exc:
            print(f"  [warn] could not delete {path}: {exc}", file=sys.stderr)
    foreign = sum(
        1
        for entry in os.listdir(folder)
        if entry.lower().endswith(IMAGE_EXTS)
        and os.path.isfile(os.path.join(folder, entry))
    )
    return removed, foreign


class FrameSchedule:
    """Decide which frames to keep so the kept ones are evenly spaced in *time*.

    Sampling by "every Nth frame" has two failure modes this replaces: N is an
    integer, so 20 fps out of a 29.97 fps clip rounds to N=1 and silently keeps
    every frame; and on variable-frame-rate video (most phones) a fixed frame
    stride makes the real time between kept frames swing wildly.
    """

    def __init__(self, rate_fps: float, tolerance_ms: float = 0.0):
        self.step_ms = 1000.0 / max(1e-6, rate_fps)
        # Container timestamps are quantised (whole milliseconds, usually), so
        # an exact ">= target" test drifts: a frame 0.4 ms short of its target
        # is rejected and the next one lands early, which shows up as an
        # irregular 1-2-1-2 stride. Half a frame either way makes the test
        # "nearest frame to the target" instead.
        self.tolerance_ms = tolerance_ms
        self.next_ms = 0.0

    def wants(self, timestamp_ms: float) -> bool:
        reach = timestamp_ms + self.tolerance_ms
        if reach < self.next_ms:
            return False
        # Advance past every target the source cannot satisfy. Without this a
        # long gap in a VFR clip is followed by a burst of consecutive frames
        # "catching up" on missed targets.
        while self.next_ms <= reach:
            self.next_ms += self.step_ms
        return True


def timestamps_usable(cap, rewind_to: int, probe: int = 30) -> bool:
    """Can this file's frame timestamps be trusted to drive the schedule?

    Most files: yes, and sampling by time is what makes a requested rate come
    out right on 29.97 fps and variable-frame-rate clips alike. Some files
    report 0 for every frame, and sampling those by time would keep exactly one
    frame -- so probe a few frames first and fall back to counting instead.
    """
    seen: list[float] = []
    for _ in range(probe):
        seen.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        if not cap.grab():
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, rewind_to)
    if len(seen) < 3 or seen[-1] <= seen[0]:
        return False
    advancing = sum(1 for before, after in zip(seen, seen[1:]) if after > before)
    return advancing >= (len(seen) - 1) / 2


def cli_extract(
    input_vid: str,
    output_folder: str,
    sample_rate: int,
    rotation: int,
    jpg_q: int,
    resolution: str,
    custom_width: int,
    trim_start: float,
    trim_end: float,
    keep_existing: bool = False,
) -> int:
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(input_vid)
    if not cap.isOpened():
        print(f"ERROR: could not open video file {input_vid}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not fps or fps <= 0 or total_frames <= 0 or orig_w <= 0:
        print(
            f"ERROR: {input_vid} reports fps={fps} frames={total_frames} "
            f"size={orig_w}x{orig_h} -- the file looks unreadable or corrupt.",
            file=sys.stderr,
        )
        cap.release()
        return 1

    total_duration = total_frames / fps

    start_frame = max(0, int(trim_start * fps))
    end_frame = total_frames - int(trim_end * fps)
    end_frame = min(total_frames, max(start_frame + int(fps), end_frame))
    if start_frame >= end_frame:
        print(
            f"ERROR: trim leaves no frames (start {trim_start}s, end {trim_end}s, "
            f"clip is {total_duration:.1f}s)",
            file=sys.stderr,
        )
        cap.release()
        return 1

    effective_frames = end_frame - start_frame
    effective_duration = effective_frames / fps

    if resolution == "custom":
        target_w = max(32, custom_width)
    else:
        target_w = max(32, orig_w // SCALE_DIVISORS.get(resolution, 1))
    target_h = max(1, int(orig_h * (target_w / orig_w)))

    if keep_existing:
        print("Keeping whatever is already in the output folder (--keep-existing).")
    else:
        removed, foreign = clear_previous_frames(output_folder)
        if removed:
            print(f"Removed {removed} frame(s) from a previous extraction.")
        if foreign:
            print(
                f"  [warn] {foreign} other image(s) are already in '{output_folder}'. "
                f"They are not named frame_*.jpg so they were left in place -- they "
                f"will still be fed to SfM alongside the new frames."
            )

    trim_info = ""
    if trim_start > 0 or trim_end > 0:
        trim_info = (
            f" (trimmed -{trim_start:.1f}s start, -{trim_end:.1f}s end, "
            f"using {effective_duration:.1f}s)"
        )
    expected = max(1, math.ceil(effective_duration * sample_rate))
    print(
        f"Extracting ~{min(expected, effective_frames)} frames from a "
        f"{total_duration:.1f}s video at {sample_rate} fps{trim_info}"
    )
    print(f"Resolution: {orig_w}x{orig_h} -> {target_w}x{target_h}")
    if sample_rate > fps:
        print(
            f"  [warn] the source is only {fps:.2f} fps, so every frame is kept; "
            f"asking for {sample_rate} fps cannot produce more."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    nominal_step_ms = 1000.0 / fps
    by_time = timestamps_usable(cap, start_frame)
    schedule = FrameSchedule(sample_rate, tolerance_ms=nominal_step_ms / 2.0)
    end_offset_ms = effective_frames * nominal_step_ms
    origin_ms: float | None = None
    elapsed_ms = 0.0
    last_ms: float | None = None
    current = start_frame
    extracted = 0
    truncated = False

    while current < end_frame:
        # Read the timestamp before read(): it reports the position of the
        # frame that read() is about to return.
        raw_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, frame = cap.read()
        if not ok:
            truncated = current < end_frame - 1
            break

        if by_time:
            if origin_ms is None:
                origin_ms = raw_ms if raw_ms and raw_ms > 0 else 0.0
            elapsed_ms = raw_ms - origin_ms
            if last_ms is not None and elapsed_ms <= last_ms:
                # A repeated timestamp means a repeated instant, not elapsed
                # time: clamp so the duplicate is skipped rather than inventing
                # time and running the schedule ahead of the clip.
                elapsed_ms = last_ms
        else:
            elapsed_ms = (current - start_frame) * nominal_step_ms
        last_ms = elapsed_ms

        if trim_end > 0 and elapsed_ms >= end_offset_ms:
            break

        if schedule.wants(elapsed_ms):
            if resolution != "original" or target_w != orig_w:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if rotation in ROTATIONS:
                frame = cv2.rotate(frame, ROTATIONS[rotation])
            out_path = os.path.join(output_folder, frame_name(extracted))
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_q])
            extracted += 1
        current += 1
    cap.release()

    if truncated:
        print(
            f"  [warn] decoding stopped at frame {current} of {end_frame} -- the "
            f"file may be truncated or damaged. Keeping what was extracted.",
            file=sys.stderr,
        )

    if extracted == 0:
        print("ERROR: no frames were extracted", file=sys.stderr)
        return 1

    achieved = extracted / (elapsed_ms / 1000.0) if elapsed_ms > 0 else float(sample_rate)
    rotated = f", rotated {rotation}deg" if rotation else ""
    print(
        f"Extraction complete: {extracted} frames saved to '{output_folder}' "
        f"({achieved:.2f} fps effective, {target_w}x{target_h}{rotated}, quality {jpg_q})"
    )
    return 0


def cli_filter(
    input_dir: str,
    output_dir: str | None,
    exts: tuple[str, ...],
    target_count: int | None,
    target_pct: float | None,
    mode: str,
    groups: int | None,
    scalar: int | None,
    pretend: bool,
    group_size: int | None = None,
) -> int:
    in_place = not bool(output_dir)
    target_folder = output_dir or input_dir

    images: list[str] = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
    images.sort()
    total = len(images)
    print(f"Found {total} images in '{input_dir}'")
    if total == 0:
        print(f"ERROR: no images matching {exts} in {input_dir}", file=sys.stderr)
        return 1

    if target_count is None:
        target_count = max(1, int(total * ((target_pct or 100) / 100)))
    target_count = max(1, min(target_count, total))

    if target_count == total:
        print("Target equals the number of available images -- nothing to filter.")
        return 0

    print(
        f"Filtering: keeping {target_count}/{total} images "
        f"({target_count / total:.1%}), mode={mode}"
    )

    selector = ImageSelector(images)
    if mode == "quality":
        sel_groups, sel_scalar = 1, None
    elif mode == "balanced":
        # group_size is the plain-language form: how many consecutive frames
        # compete against each other. scalar is the legacy spelling.
        sel_scalar = None if group_size else (scalar or 1)
        sel_groups = group_count_for(total, group_size) if group_size else None
    else:  # custom
        sel_groups, sel_scalar = groups or None, None
    selected = selector.filter_sharpest_images(target_count, sel_groups, sel_scalar)

    if pretend:
        print(f"Pretend: would keep {len(selected)} images")
        return 0

    if in_place:
        keep = set(selected)
        removed = 0
        for img in images:
            if img not in keep:
                os.remove(img)
                removed += 1
        print(
            f"Filtering complete: deleted {removed} images, "
            f"retained {len(selected)} in place."
        )
    else:
        pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)
        for img in selected:
            shutil.copy2(img, os.path.join(target_folder, os.path.basename(img)))
        print(f"Filtering complete: {len(selected)} images copied to '{target_folder}'")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gs-recon frames",
        description="Extract frames from video and filter them by sharpness",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ex = sub.add_parser("extract", help="Extract frames from a video")
    p_ex.add_argument("--input", required=True, help="Input video file")
    p_ex.add_argument("--output", required=True, help="Output folder for frames")
    p_ex.add_argument("--sample-rate", type=int, default=15, help="Sample rate in fps")
    p_ex.add_argument("--rotation", type=int, choices=[0, -90, 90, 180], default=0)
    p_ex.add_argument("--jpg-q", type=int, default=100, help="JPEG quality (1-100)")
    p_ex.add_argument(
        "--resolution",
        choices=["original", "half", "quarter", "eighth", "custom"],
        default="original",
    )
    p_ex.add_argument("--width", type=int, default=1280, help="Width when --resolution custom")
    p_ex.add_argument("--trim-start", type=float, default=0, help="Seconds to drop from the start")
    p_ex.add_argument("--trim-end", type=float, default=0, help="Seconds to drop from the end")
    p_ex.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete frames left in the output folder by a previous run "
             "(they would be mixed into this run's sequence)",
    )

    p_f = sub.add_parser("filter", help="Keep only the sharpest frames")
    p_f.add_argument("--input", required=True, help="Folder of frames")
    p_f.add_argument("--output", help="Copy selection here (omit to filter in place)")
    p_f.add_argument("--mode", choices=["quality", "balanced", "custom"], default="balanced")
    p_f.add_argument("--exts", default="jpg,jpeg,png", help="Comma-separated extensions")
    target = p_f.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-count", type=int, help="Number of images to retain")
    target.add_argument("--target-percentage", type=float, help="Percentage to retain")
    div = p_f.add_mutually_exclusive_group()
    div.add_argument("--groups", type=int, help="Group count (mode=custom)")
    div.add_argument(
        "--group-size", type=int,
        help="Frames per group in mode=balanced: the sharpest of every N "
             "consecutive frames are kept",
    )
    div.add_argument("--scalar", type=int, help="Legacy spelling of --group-size (mode=balanced)")
    p_f.add_argument("--pretend", action="store_true", help="Report the selection, change nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        return cli_extract(
            args.input, args.output, args.sample_rate, args.rotation, args.jpg_q,
            args.resolution, args.width, args.trim_start, args.trim_end,
            args.keep_existing,
        )
    exts = tuple(ext.strip().lower() for ext in args.exts.split(",") if ext.strip())
    return cli_filter(
        args.input, args.output, exts, args.target_count, args.target_percentage,
        args.mode, args.groups, args.scalar, args.pretend, args.group_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
