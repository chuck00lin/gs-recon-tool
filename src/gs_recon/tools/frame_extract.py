"""Extract video frames and cull them down to the sharpest subset.

Usable standalone::

    python -m gs_recon.tools.frame_extract extract --input clip.mp4 --output ./images
    python -m gs_recon.tools.frame_extract filter  --input ./images --target-percentage 20
"""

from __future__ import annotations

import argparse
import glob
import os
import pathlib
import shutil
import sys

import cv2

from .image_selector import ImageSelector

ROTATIONS = {
    -90: cv2.ROTATE_90_COUNTERCLOCKWISE,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
}
SCALE_DIVISORS = {"original": 1, "half": 2, "quarter": 4, "eighth": 8}


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

    interval = max(1, int(round(fps / max(1, sample_rate))))
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

    trim_info = ""
    if trim_start > 0 or trim_end > 0:
        trim_info = (
            f" (trimmed -{trim_start:.1f}s start, -{trim_end:.1f}s end, "
            f"using {effective_duration:.1f}s)"
        )
    print(
        f"Extracting ~{effective_frames // interval} frames from a "
        f"{total_duration:.1f}s video at {sample_rate} fps{trim_info}"
    )
    print(f"Resolution: {orig_w}x{orig_h} -> {target_w}x{target_h}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    extracted = 0
    while current < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (current - start_frame) % interval == 0:
            if resolution != "original" or target_w != orig_w:
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if rotation in ROTATIONS:
                frame = cv2.rotate(frame, ROTATIONS[rotation])
            out_path = os.path.join(output_folder, f"frame_{extracted:04d}.jpg")
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_q])
            extracted += 1
        current += 1
    cap.release()

    if extracted == 0:
        print("ERROR: no frames were extracted", file=sys.stderr)
        return 1

    rotated = f", rotated {rotation}deg" if rotation else ""
    print(
        f"Extraction complete: {extracted} frames saved to '{output_folder}' "
        f"({sample_rate} fps sampling, {target_w}x{target_h}{rotated}, quality {jpg_q})"
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
        sel_groups, sel_scalar = None, scalar or 1
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
    div.add_argument("--scalar", type=int, help="Scalar (mode=balanced)")
    p_f.add_argument("--pretend", action="store_true", help="Report the selection, change nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract":
        return cli_extract(
            args.input, args.output, args.sample_rate, args.rotation, args.jpg_q,
            args.resolution, args.width, args.trim_start, args.trim_end,
        )
    exts = tuple(ext.strip().lower() for ext in args.exts.split(",") if ext.strip())
    return cli_filter(
        args.input, args.output, exts, args.target_count, args.target_percentage,
        args.mode, args.groups, args.scalar, args.pretend,
    )


if __name__ == "__main__":
    raise SystemExit(main())
