"""Prune a 3D Gaussian Splatting PLY by opacity and radial distance.

Trained splats carry a long tail of near-transparent Gaussians and a halo of
far-flung outliers seeded by SfM noise. Both cost memory in every downstream
viewer and neither contributes to the render, so they are dropped here.

The file is rewritten in place-compatible binary-little-endian form: the header
is copied verbatim apart from the vertex count, so any properties beyond the
ones we inspect survive untouched.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_header(f) -> tuple[list[bytes], int, list[str]]:
    header_lines: list[bytes] = []
    num_vertices: int | None = None
    prop_names: list[str] = []
    in_vertex = False

    while True:
        line = f.readline()
        if not line:
            raise RuntimeError("Unexpected EOF before end_header")
        header_lines.append(line)
        stripped = line.strip()

        if stripped.startswith(b"element"):
            parts = stripped.split()
            in_vertex = len(parts) >= 3 and parts[1] == b"vertex"
            if in_vertex:
                num_vertices = int(parts[2])
            continue

        if in_vertex and stripped.startswith(b"property"):
            prop_names.append(stripped.split()[-1].decode("ascii"))

        if stripped == b"end_header":
            break

    if num_vertices is None:
        raise RuntimeError("Could not find 'element vertex' in header")
    return header_lines, num_vertices, prop_names


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def prune(
    input_path: str,
    output_path: str,
    alpha_threshold: float = 0.005,
    radial_percentile: float = 99.0,
    radial_margin: float = 1.0,
) -> int:
    with open(input_path, "rb") as f:
        header_lines, num_vertices, prop_names = parse_header(f)
        missing = [p for p in ("opacity", "x", "y", "z") if p not in prop_names]
        if missing:
            raise RuntimeError(
                f"{input_path}: vertex element is missing {missing}. "
                f"This does not look like a Gaussian-splat PLY."
            )
        opacity_idx = prop_names.index("opacity")
        xyz_idx = [prop_names.index(a) for a in ("x", "y", "z")]

        num_props = len(prop_names)
        vertices_raw = f.read(num_vertices * 4 * num_props)
        rest_data = f.read()

    verts = np.frombuffer(vertices_raw, dtype="<f4").reshape(num_vertices, num_props)

    alpha = sigmoid(verts[:, opacity_idx])
    alpha_mask = alpha >= alpha_threshold

    coords = verts[:, xyz_idx]
    if not np.any(alpha_mask):
        print(
            f"WARNING: no Gaussian has alpha >= {alpha_threshold}; "
            f"the threshold is probably too high.",
            file=sys.stderr,
        )
        keep_mask = alpha_mask
    else:
        centre = np.median(coords[alpha_mask], axis=0)
        radius = np.linalg.norm(coords - centre[None, :], axis=1)
        r_max = np.percentile(radius[alpha_mask], radial_percentile) * radial_margin
        keep_mask = alpha_mask & (radius <= r_max)

    kept = verts[keep_mask]
    new_count = int(kept.shape[0])

    print(f"Original vertices: {num_vertices}")
    print(f"After alpha >= {alpha_threshold}: {int(alpha_mask.sum())}")
    print(f"After radius filter (p{radial_percentile} x {radial_margin}): {new_count}")
    print(f"Removed total: {num_vertices - new_count}")

    new_header: list[bytes] = []
    for line in header_lines:
        stripped = line.strip()
        parts = stripped.split()
        if stripped.startswith(b"element") and len(parts) >= 3 and parts[1] == b"vertex":
            parts[2] = str(new_count).encode("ascii")
            line = b" ".join(parts) + b"\n"
        new_header.append(line)

    with open(output_path, "wb") as f:
        for line in new_header:
            f.write(line)
        f.write(kept.astype("<f4").tobytes())
        f.write(rest_data)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gs-recon prune",
        description="Prune a Gaussian-splat PLY by opacity and radial distance",
    )
    parser.add_argument("input", help="Input .ply")
    parser.add_argument("output", nargs="?", help="Output .ply (default: overwrite input)")
    parser.add_argument("--alpha", type=float, default=0.005, help="Minimum alpha to keep")
    parser.add_argument("--percentile", type=float, default=99.0, help="Radius percentile")
    parser.add_argument("--margin", type=float, default=1.0, help="Radius percentile multiplier")
    args = parser.parse_args(argv)
    return prune(
        args.input, args.output or args.input,
        args.alpha, args.percentile, args.margin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
